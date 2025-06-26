import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
import yaml
from autoattack import AutoAttack
from torch.nn import KLDivLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchattacks import PGD, FGSM

from utils import CustomCrossEntropyLoss
from utils import set_seed, Logger, get_loader, get_test_loader, estimate_total_flops

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'], help='')
parser.add_argument('--seed', type=int, default=0, help='random seed')
argparse = parser.parse_args()
set_seed(argparse.seed)

with open('configs/configs.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

configs = configs[argparse.dataset]

eps = configs['eps'] / 255
device = torch.device(configs['device'] if torch.cuda.is_available() else "cpu")


train_loader, val_loader = get_loader(argparse.dataset, configs)


if argparse.dataset == 'cifar10' or argparse.dataset == 'cifar100':
    from models import ResNet18

    model = ResNet18(num_classes=configs['class_num']).to(device)
elif argparse.dataset == 'tiny-imagenet':
    from models import PreActResNet18

    model = PreActResNet18(num_classes=configs['class_num']).to(device)

wa_model = copy.deepcopy(model)
exp_avg = model.state_dict()

opt = SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=5e-4)
scheduler = OneCycleLR(opt, max_lr=configs['max_lr'], total_steps=configs['epochs'])

criterion = CustomCrossEntropyLoss(label_smoothing=configs['ls'])
loss_fn = KLDivLoss(reduction='batchmean')
pgd_attacker = PGD(wa_model, eps=eps, alpha=2.0 / 255, steps=10)
fgsm_attacker = FGSM(wa_model, eps=eps)

output_dir = os.path.join(configs['log_path'],
                          time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-seed-' + str(argparse.seed))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'models')):
    os.makedirs(os.path.join(output_dir, 'models'))

logger_path = os.path.join(output_dir, 'output.log')
logger = Logger(logger_path)

total_training_time, total_val_time = 0, 0
total_forward_num, total_backward_num = 0, 0
train_acc_list, val_acc_list, val_pgd_acc_list, val_fgsm_acc_list = [], [], [], []
best_acc = 0
cur_lambda, cur_update = 0, configs['lambda'] / configs['epochs']

for epoch in range(configs['epochs']):
    logger.log('============ Epoch {} ============'.format(epoch))
    start_time = time.time()

    cur_lambda += cur_update
    model.train()

    train_correct, train_num = 0, 0
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        if idx % configs['trap_update'] == 0:
            model.eval()
            constant_input = torch.full((configs['class_num'], 3, configs['image_size'], configs['image_size']),
                                        0.5).to(device)
            constant_input.requires_grad_(True)
            sequential_labels = torch.arange(configs['class_num']).to(device)

            output = model(constant_input)
            loss = criterion(output, sequential_labels)
            loss.backward()
            trap = eps * torch.sign(constant_input.grad).detach()

            total_forward_num += configs['class_num']
            total_backward_num += configs['class_num']

            model.train()

        delta = trap[labels].clone()
        bonulli_idx = torch.rand_like(images)
        uniform_noise = torch.zeros_like(images).uniform_(-eps, eps)
        mask_ratio = torch.rand(1).item() * (configs['max_mask_ratio'] - configs['min_mask_ratio']) + configs[
            'min_mask_ratio']
        delta[bonulli_idx < mask_ratio] = uniform_noise[bonulli_idx < mask_ratio]

        if torch.rand(1) < 0.5:
            adv_images = torch.clamp(images.clone() - delta, 0, 1).detach()
        else:
            adv_images = torch.clamp(images.clone() + delta, 0, 1).detach()

        adv_images = adv_images.detach().requires_grad_(True)

        ori_output = model(adv_images)
        ori_loss = criterion(ori_output, labels)
        ori_loss.backward(retain_graph=True)
        delta = eps * adv_images.grad.sign().detach()

        total_forward_num += adv_images.shape[0]
        total_backward_num += adv_images.shape[0]

        adv_images = adv_images + delta
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        output = model(adv_images)
        train_correct += (output.argmax(1) == labels).sum().item()
        train_num += images.shape[0]

        loss1 = criterion(output, labels)
        p = F.softmax(ori_output, dim=1)
        q = F.softmax(output, dim=1)
        m = torch.clamp((p + q) / 2., 0, 1).log()
        loss2 = cur_lambda * (F.kl_div(m, p, reduction='batchmean') + F.kl_div(m, q, reduction='batchmean')) / 2

        loss = loss1 + loss2

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_forward_num += adv_images.shape[0]
        total_backward_num += adv_images.shape[0]

        for key, value in model.state_dict().items():
            exp_avg[key] = (1 - configs['tau']) * value + configs['tau'] * exp_avg[key]

    scheduler.step()

    end_time = time.time()
    total_training_time += end_time - start_time
    logger.log('Train Acc: {:.4f}'.format(train_correct / train_num))
    logger.log('Train Time: {:.4f}'.format(end_time - start_time))
    train_acc_list.append(train_correct / train_num)

    # Validation
    wa_model.load_state_dict(exp_avg)
    wa_model.eval()
    model.eval()
    val_correct, fgsm_correct, pgd_correct, val_num = 0, 0, 0, 0
    start_time = time.time()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        output = wa_model(images)
        val_correct += (output.argmax(1) == labels).sum().item()
        val_num += images.shape[0]

        fgsm_images = fgsm_attacker(images, labels)
        output = wa_model(fgsm_images)
        fgsm_correct += (output.argmax(1) == labels).sum().item()

        pgd_images = pgd_attacker(images, labels)
        output = wa_model(pgd_images)
        pgd_correct += (output.argmax(1) == labels).sum().item()

    end_time = time.time()
    total_val_time += end_time - start_time
    logger.log('Val Acc: {:.4f}'.format(val_correct / val_num))
    logger.log('VAL FGSM Acc: {:.4f}'.format(fgsm_correct / val_num))
    logger.log('VAL PGD Acc: {:.4f}'.format(pgd_correct / val_num))
    logger.log('Val Time: {:.4f}'.format(end_time - start_time))

    val_acc_list.append(val_correct / val_num)
    val_fgsm_acc_list.append(fgsm_correct / val_num)
    val_pgd_acc_list.append(pgd_correct / val_num)

    if pgd_correct / val_num >= best_acc:
        best_acc = pgd_correct / val_num
        torch.save(wa_model.state_dict(), os.path.join(output_dir, 'models', 'best.pth'))

    if epoch == configs['epochs'] - 1:
        torch.save(wa_model.state_dict(), os.path.join(output_dir, 'models', 'last.pth'.format(epoch)))
        torch.save(model.state_dict(), os.path.join(output_dir, 'models', 'ori-last.pth'.format(epoch)))

logger.new_line()

logger.log('Train acc list: {}'.format(train_acc_list))
logger.log('Val acc list: {}'.format(val_acc_list))
logger.log('Val FGSM acc list: {}'.format(val_fgsm_acc_list))
logger.log('Val PGD acc list: {}'.format(val_pgd_acc_list))
logger.log('Total training time: {:.4f}'.format(total_training_time))
logger.log('Total validation time: {:.4f}'.format(total_val_time))
flops = estimate_total_flops(model, (1, 3, configs['image_size'], configs['image_size']),
                             forward_passes=total_forward_num, backward_passes=total_backward_num, device=device)
# PetaFLOPs
logger.log('Total FLOPs: {:.4f} PetaFLOPs'.format(flops / 10 ** 15))
logger.log('\n')
logger.log('\n')


# test
test_loader = get_test_loader(argparse.dataset, configs)

model.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'best.pth')))
model.eval()

# PGD
steps = [10, 20, 50]
for step in steps:
    attacker = PGD(model, eps=eps, alpha=2.0 / 255, steps=step)
    clean_correct, pgd_correct, total = 0, 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if step == steps[0]:
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            clean_correct += (pre == labels).sum().item()

        adv_images = attacker(images, labels)
        outputs = model(adv_images)
        _, pre = torch.max(outputs.data, 1)
        pgd_correct += (pre == labels).sum().item()
        total += labels.size(0)
    if step == steps[0]:
        logger.log('clean accuracy: {} %'.format(100 * clean_correct / total))
    logger.log('pgd-{} accuracy: {} %'.format(step, 100 * pgd_correct / total))


# AutoAttack
logger.new_line()
test_loader = get_test_loader(argparse.dataset, configs, batch_size=10000)

X, y = [], []
for i, (x, y_) in enumerate(test_loader):
    X = x.to(device)
    y = y_.to(device)

attacker = AutoAttack(model, eps=eps, device=device, log_path=logger_path)
x_adv = attacker.run_standard_evaluation(X, y, bs=200)
