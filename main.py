# Import necessary libraries
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

# Import custom utility functions
from utils import CustomCrossEntropyLoss
from utils import set_seed, Logger, get_loader, get_test_loader, estimate_total_flops

# --- Argument Parsing & Initial Setup ---

# Setup argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'tiny-imagenet', 'cifar10-eps16', 'cifar100-eps16',
                             'tiny-imagenet-eps12'], help='The dataset to use for training and evaluation.')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
argparse = parser.parse_args()

# Set the random seed for all random number generators to ensure reproducibility
set_seed(argparse.seed)

# Load training configurations from a YAML file
with open('configs/configs.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

# Select the configuration specific to the chosen dataset
configs = configs[argparse.dataset]

# Define the perturbation budget (epsilon) and normalize it
eps = configs['eps'] / 255
# Set the device for computation (GPU if available, otherwise CPU)
device = torch.device(configs['device'] if torch.cuda.is_available() else "cpu")

# --- Data, Model, and Optimizer Setup ---

# Get data loaders for the training and validation sets
train_loader, val_loader = get_loader(argparse.dataset, configs)

# Initialize the model architecture based on the dataset
if argparse.dataset == 'cifar10' or argparse.dataset == 'cifar100':
    from models import ResNet18

    model = ResNet18(num_classes=configs['class_num']).to(device)
elif argparse.dataset == 'tiny-imagenet':
    from models import PreActResNet18

    model = PreActResNet18(num_classes=configs['class_num']).to(device)

# Create a weight-averaged (WA) model for evaluation, which often improves robustness
wa_model = copy.deepcopy(model)
# Initialize the exponential moving average (EMA) of model weights
exp_avg = model.state_dict()

# Initialize the SGD optimizer
opt = SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=5e-4)
# Initialize the OneCycleLR learning rate scheduler
scheduler = OneCycleLR(opt, max_lr=configs['max_lr'], total_steps=configs['epochs'])

# --- Loss Functions and Attackers ---

# Define custom cross-entropy loss with label smoothing
criterion = CustomCrossEntropyLoss(label_smoothing=configs['ls'])
# Define KL-Divergence loss for consistency regularization
loss_fn = KLDivLoss(reduction='batchmean')
# Initialize PGD and FGSM attackers for validation
pgd_attacker = PGD(wa_model, eps=eps, alpha=2.0 / 255, steps=10)
fgsm_attacker = FGSM(wa_model, eps=eps)

# --- Logging and Output Setup ---

# Setup output directory for logs and models, named with a timestamp and seed
output_dir = os.path.join(configs['log_path'],
                          time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-seed-' + str(argparse.seed))

# Create directories if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'models')):
    os.makedirs(os.path.join(output_dir, 'models'))

# Initialize a logger to save console output to a file
logger_path = os.path.join(output_dir, 'output.log')
logger = Logger(logger_path)

# --- Training Initialization ---

# Initialize variables for tracking metrics and training state
total_training_time, total_val_time = 0, 0
total_forward_num, total_backward_num = 0, 0
train_acc_list, val_acc_list, val_pgd_acc_list, val_fgsm_acc_list = [], [], [], []
best_acc = 0
# Initialize lambda for consistency loss, which will be increased linearly during training
cur_lambda, cur_update = 0, configs['lambda'] / configs['epochs']

# --- Main Training Loop ---

for epoch in range(configs['epochs']):
    logger.log('============ Epoch {} ============'.format(epoch))
    start_time = time.time()

    # Linearly increase the weight of the consistency loss term
    cur_lambda += cur_update
    # Set the model to training mode
    model.train()

    train_correct, train_num = 0, 0
    # Iterate over batches of training data
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # --- Label-Indicative Perturbation (LI) Generation ---
        # Periodically generate the "transferable adversarial perturbation" or "trap"
        if idx % configs[('li_update')] == 0:
            model.eval()  # Use eval mode to get clean gradients without batch norm updates
            # Create a batch of constant gray images, one for each class
            constant_input = torch.full((configs['class_num'], 3, configs['image_size'], configs['image_size']),
                                        0.5).to(device)
            constant_input.requires_grad_(True)
            # Create labels from 0 to num_classes-1 to target each class
            sequential_labels = torch.arange(configs['class_num']).to(device)

            # Forward pass and compute loss for the constant inputs
            output = model(constant_input)
            loss = criterion(output, sequential_labels)
            loss.backward()
            # Generate the Label-Indicative (LI) perturbation by taking the sign of the gradients
            LI = eps * torch.sign(constant_input.grad).detach()

            # Track FLOPs
            total_forward_num += configs['class_num']
            total_backward_num += configs['class_num']

            model.train()  # Return model to training mode

        # --- Perturbation Crafting (Label Information Elimination) ---
        # Select the pre-computed LI corresponding to the labels of the current batch
        delta = LI[labels].clone()
        # Generate a random binary mask
        bonulli_idx = torch.rand_like(images)
        # Create uniform noise to replace parts of the perturbation
        if configs['eps'] > 8:  # Use a larger noise range for larger epsilon
            uniform_noise = torch.zeros_like(images).uniform_(-2 * eps, 2 * eps)
        else:
            uniform_noise = torch.zeros_like(images).uniform_(-eps, eps)
        # Determine a random ratio for the mask
        mask_ratio = torch.rand(1).item() * (configs['max_mask_ratio'] - configs['min_mask_ratio']) + configs[
            'min_mask_ratio']
        # Apply the mask: replace parts of the LI with uniform noise to "eliminate" label information
        delta[bonulli_idx < mask_ratio] = uniform_noise[bonulli_idx < mask_ratio]

        # Create initial adversarial images by applying the crafted delta
        if torch.rand(1) < 0.5:  # Randomly choose to add or subtract the perturbation
            adv_images = torch.clamp(images.clone() - delta, 0, 1).detach()
        else:
            adv_images = torch.clamp(images.clone() + delta, 0, 1).detach()

        # --- PGD-1 Refinement and Consistency Regularization ---
        adv_images = adv_images.detach().requires_grad_(True)

        # Get the model's output on these initial adversarial images
        ori_output = model(adv_images)
        ori_loss = criterion(ori_output, labels)
        # Calculate gradients w.r.t. the initial adversarial images
        ori_loss.backward(retain_graph=True)
        # Perform one PGD step to refine the perturbation
        delta = eps * adv_images.grad.sign().detach()

        # Track FLOPs
        total_forward_num += adv_images.shape[0]
        total_backward_num += adv_images.shape[0]

        # Create the final adversarial images
        adv_images = adv_images + delta
        # Clamp the final images to the valid [0, 1] range and ensure the perturbation is within the epsilon-ball
        if configs['eps'] > 8:
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        # --- Loss Calculation and Optimization ---
        # Get the model's output on the final adversarial images
        output = model(adv_images)
        train_correct += (output.argmax(1) == labels).sum().item()
        train_num += images.shape[0]

        # Loss 1: Standard adversarial training loss
        loss1 = criterion(output, labels)
        # Loss 2: Consistency regularization loss (JSD-like)
        p = F.softmax(ori_output, dim=1)
        q = F.softmax(output, dim=1)
        m = torch.clamp((p + q) / 2., 0, 1).log()
        # The loss encourages the model to produce similar outputs for the initial and refined adversarial examples
        loss2 = cur_lambda * (F.kl_div(m, p, reduction='batchmean') + F.kl_div(m, q, reduction='batchmean')) / 2

        # Combine the two losses
        loss = loss1 + loss2

        # Standard optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Track FLOPs
        total_forward_num += adv_images.shape[0]
        total_backward_num += adv_images.shape[0]

        # Update the exponential moving average of the model weights
        for key, value in model.state_dict().items():
            exp_avg[key] = (1 - configs['tau']) * value + configs['tau'] * exp_avg[key]

    # Step the learning rate scheduler
    scheduler.step()

    # --- End of Epoch Logging ---
    end_time = time.time()
    total_training_time += end_time - start_time
    logger.log('Train Acc: {:.4f}'.format(train_correct / train_num))
    logger.log('Train Time: {:.4f}'.format(end_time - start_time))
    train_acc_list.append(train_correct / train_num)

    # --- Validation ---
    # Load the averaged weights into the evaluation model
    wa_model.load_state_dict(exp_avg)
    wa_model.eval()
    model.eval()
    val_correct, fgsm_correct, pgd_correct, val_num = 0, 0, 0, 0
    start_time = time.time()
    # Iterate over the validation set
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        # Evaluate clean accuracy
        output = wa_model(images)
        val_correct += (output.argmax(1) == labels).sum().item()
        val_num += images.shape[0]

        # Evaluate robust accuracy against FGSM
        fgsm_images = fgsm_attacker(images, labels)
        output = wa_model(fgsm_images)
        fgsm_correct += (output.argmax(1) == labels).sum().item()

        # Evaluate robust accuracy against PGD
        pgd_images = pgd_attacker(images, labels)
        output = wa_model(pgd_images)
        pgd_correct += (output.argmax(1) == labels).sum().item()

    end_time = time.time()
    total_val_time += end_time - start_time
    # Log validation metrics
    logger.log('Val Acc: {:.4f}'.format(val_correct / val_num))
    logger.log('VAL FGSM Acc: {:.4f}'.format(fgsm_correct / val_num))
    logger.log('VAL PGD Acc: {:.4f}'.format(pgd_correct / val_num))
    logger.log('Val Time: {:.4f}'.format(end_time - start_time))

    val_acc_list.append(val_correct / val_num)
    val_fgsm_acc_list.append(fgsm_correct / val_num)
    val_pgd_acc_list.append(pgd_correct / val_num)

    # --- Model Checkpointing ---
    # If current PGD accuracy is the best so far, save the WA model
    if pgd_correct / val_num >= best_acc:
        best_acc = pgd_correct / val_num
        torch.save(wa_model.state_dict(), os.path.join(output_dir, 'models', 'best.pth'))

    # Save the final models at the end of training
    if epoch == configs['epochs'] - 1:
        torch.save(wa_model.state_dict(), os.path.join(output_dir, 'models', 'last.pth'))
        torch.save(model.state_dict(), os.path.join(output_dir, 'models', 'ori-last.pth'))

# --- Final Logging ---
logger.new_line()

# Log final results and statistics after training is complete
logger.log('Train acc list: {}'.format(train_acc_list))
logger.log('Val acc list: {}'.format(val_acc_list))
logger.log('Val FGSM acc list: {}'.format(val_fgsm_acc_list))
logger.log('Val PGD acc list: {}'.format(val_pgd_acc_list))
logger.log('Total training time: {:.4f}'.format(total_training_time))
logger.log('Total validation time: {:.4f}'.format(total_val_time))
# Estimate and log the total computational cost (FLOPs)
flops = estimate_total_flops(model, (1, 3, configs['image_size'], configs['image_size']),
                             forward_passes=total_forward_num, backward_passes=total_backward_num, device=device)
logger.log('Total FLOPs: {:.4f} PetaFLOPs'.format(flops / 10 ** 15))
logger.log('\n')
logger.log('\n')

# --- Final Testing on Test Set ---
# Get the test data loader
test_loader = get_test_loader(argparse.dataset, configs)

# Load the best performing model (based on validation PGD accuracy)
model.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'best.pth')))
model.eval()

# Test against PGD with various step sizes
steps = [10, 20, 50]
for step in steps:
    attacker = PGD(model, eps=eps, alpha=2.0 / 255, steps=step)
    clean_correct, pgd_correct, total = 0, 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Calculate clean accuracy only once
        if step == steps[0]:
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            clean_correct += (pre == labels).sum().item()

        # Generate and evaluate on PGD adversarial examples
        adv_images = attacker(images, labels)
        outputs = model(adv_images)
        _, pre = torch.max(outputs.data, 1)
        pgd_correct += (pre == labels).sum().item()
        total += labels.size(0)
    if step == steps[0]:
        logger.log('clean accuracy: {} %'.format(100 * clean_correct / total))
    logger.log('pgd-{} accuracy: {} %'.format(step, 100 * pgd_correct / total))

# Test against AutoAttack (a strong, parameter-free attack suite)
logger.new_line()
# Load the entire test set for AutoAttack
test_loader = get_test_loader(argparse.dataset, configs, batch_size=10000)

X, y = [], []
for i, (x, y_) in enumerate(test_loader):
    X = x.to(device)
    y = y_.to(device)

# Initialize and run AutoAttack
attacker = AutoAttack(model, eps=eps, device=device, log_path=logger_path)
x_adv = attacker.run_standard_evaluation(X, y, bs=200)
