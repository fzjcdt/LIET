import os
import time

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchattacks import PGD, FGSM
from torchvision.datasets import CIFAR10

from models import ResNet18
from utils import set_seed, Logger
from utils.dataset import train_transform, test_transform

set_seed(0)
eps = 16 / 255.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = CIFAR10('./data/cifar10', train=True, download=True, transform=train_transform)
test_set = CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

model = ResNet18(num_classes=10).to(device)
opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
scheduler = MultiStepLR(opt, milestones=[80, 90], gamma=0.1)

criterion = CrossEntropyLoss()
pgd_attacker = PGD(model, eps=eps, alpha=2.0 / 255, steps=10)
fgsm_attacker = FGSM(model, eps=eps)

output_dir = os.path.join('./log/', 'cifar10')
output_dir = os.path.join(output_dir,
                          time.strftime("%Y-%m-%d-%H-%M-%S-seed0",
                                        time.localtime()) + '-eps-' + str(int(eps * 255)))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'models')):
    os.makedirs(os.path.join(output_dir, 'models'))

logger = Logger(os.path.join(output_dir, 'output.log'))

total_training_time, total_test_time = 0.0, 0.0
train_acc_list, test_acc_list, pgd_acc_list, fgsm_acc_list = [], [], [], []

for epoch in range(100):
    logger.log('============ Epoch {} ============'.format(epoch))
    model.train()
    train_correct, train_num = 0, 0
    start_time = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        output = model(adv_images)
        loss = criterion(output, labels)
        loss.backward()

        adv_images = adv_images + eps * adv_images.grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        output = model(adv_images)
        train_correct += (output.argmax(1) == labels).sum().item()
        train_num += images.shape[0]

        loss = criterion(output, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    scheduler.step()
    end_time = time.time()
    total_training_time += end_time - start_time
    logger.log('Train Acc: {:.4f}'.format(train_correct / train_num))
    logger.log('Train Time: {:.4f}'.format(end_time - start_time))
    train_acc_list.append(train_correct / train_num)

    model.eval()
    test_correct, fgsm_correct, pgd_correct, test_num = 0, 0, 0, 0
    start_time = time.time()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        test_correct += (output.argmax(1) == labels).sum().item()
        test_num += images.shape[0]

        fgsm_images = fgsm_attacker(images, labels)
        output = model(fgsm_images)
        fgsm_correct += (output.argmax(1) == labels).sum().item()

        pgd_images = pgd_attacker(images, labels)
        output = model(pgd_images)
        pgd_correct += (output.argmax(1) == labels).sum().item()

    end_time = time.time()
    total_test_time += end_time - start_time
    logger.log('Test Acc: {:.4f}'.format(test_correct / test_num))
    logger.log('FGSM Acc: {:.4f}'.format(fgsm_correct / test_num))
    logger.log('PGD Acc: {:.4f}'.format(pgd_correct / test_num))
    logger.log('Test Time: {:.4f}'.format(end_time - start_time))

    test_acc_list.append(test_correct / test_num)
    fgsm_acc_list.append(fgsm_correct / test_num)
    pgd_acc_list.append(pgd_correct / test_num)

    torch.save(model.state_dict(), os.path.join(output_dir, 'models', 'epoch-{}.pth'.format(epoch)))

logger.log('Total Training Time: {:.4f}'.format(total_training_time))
logger.log('Total Test Time: {:.4f}'.format(total_test_time))

logger.log('Train Acc List: {}'.format(train_acc_list))
logger.log('Test Acc List: {}'.format(test_acc_list))
logger.log('FGSM Acc List: {}'.format(fgsm_acc_list))
logger.log('PGD Acc List: {}'.format(pgd_acc_list))
