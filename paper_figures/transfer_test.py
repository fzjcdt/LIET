import os
import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models import ResNet18
from utils import set_seed, Logger
from utils.dataset import test_transform

# --- Configuration ---
# Path to the log directory of a specific experiment run.
# This directory should contain the saved model checkpoints for each epoch.
experiment_path = 'log/cifar10/2024-02-02-17-11-08-seed0-eps-16/'
SEED = 0
PERTURBATION_BUDGET = 16 / 255.0  # Corresponds to ε in the paper

# --- Setup ---
set_seed(SEED)
epsilon = PERTURBATION_BUDGET
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the CIFAR-10 test dataset
test_dataset = CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=4)
total_test_samples = len(test_dataset)  # Total number of test samples N

# Define the loss function and model
criterion = CrossEntropyLoss()
num_classes = 10  # Number of classes C
model = ResNet18(num_classes=num_classes).to(device)

# Initialize logger
log_path = os.path.join(experiment_path, 'transferability_analysis.log')
logger = Logger(log_path)
logger.log(f"Starting transferability analysis for experiment: {experiment_path}")
logger.log(f"Epsilon (ε): {epsilon}, Device: {device}")

# Lists to store the final metrics for each epoch
p_abnormal_list, p_dominate_list = [], []

# --- Main Analysis Loop ---
# Iterate through each saved model checkpoint (epoch)
for epoch in range(100):
    model_path = os.path.join(experiment_path, 'models/epoch-{}.pth'.format(epoch))
    if not os.path.exists(model_path):
        logger.log(f"Model for epoch {epoch} not found, skipping.")
        continue

    model.load_state_dict(torch.load(model_path))
    model.eval()

    start_time = time.time()

    # The original code had a loop for 'sample_num', which ran only once.
    # This structure is kept in case of future extension to different base inputs (Uniform Gray (0.0)...).
    for sample_num in range(1):
        # --- Step 1: Generate Class-Specific Perturbations (δ_c) ---
        # As per the paper, we generate perturbations from a base input x.
        # Here, x is a "Uniform Gray (0.5)" image.
        # Shape: [num_classes, channels, height, width] to compute all gradients in one pass.
        base_input_x = torch.full((num_classes, 3, 32, 32), 0.5, device=device, requires_grad=True)

        # Target labels c = [0, 1, ..., C-1] for perturbation generation.
        target_labels_c = torch.arange(num_classes, device=device)

        # Compute gradients: ∇_x L(f(x;θ), c)
        logits = model(base_input_x)
        loss = criterion(logits, target_labels_c)
        loss.backward()

        # Calculate class-specific perturbations using the formula: δ_c = ε * sign(∇_x L)
        class_perturbations_delta = epsilon * torch.sign(base_input_x.grad)

        # --- Step 2: Evaluate Transferability on the Test Set ---
        # Initialize counters for the metrics' numerators.
        abnormal_count = 0
        dominate_count = 0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for test_images_xi, _ in test_loader:  # test_labels are not needed for this analysis
                test_images_xi = test_images_xi.to(device)

                # Get original model output f(x_i)
                original_logits = model(test_images_xi)

                # Apply each class perturbation δ_c to all test images x_i
                for c in range(num_classes):
                    # Get the perturbation δ_c for the current class c
                    delta_c = class_perturbations_delta[c].unsqueeze(0)

                    # Apply perturbation: x_i' = x_i + δ_c
                    perturbed_images = test_images_xi + delta_c
                    perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Project back to valid image range

                    # Get perturbed model output f(x_i')
                    perturbed_logits = model(perturbed_images)

                    # --- Calculate Metrics based on Paper's Formulas ---
                    # P_abnormal: check if f(x_i + δ_c)_c > f(x_i)_c
                    abnormal_count += torch.sum(perturbed_logits[:, c] > original_logits[:, c]).item()

                    # P_dominate: check if argmax f(x_i + δ_c) = c
                    perturbed_predictions = perturbed_logits.argmax(1)
                    dominate_count += torch.sum(perturbed_predictions == c).item()

        # Normalize the counts to get the final probabilities (as defined in the paper)
        # Denominator is N * C (total_test_samples * num_classes)
        p_abnormal = abnormal_count / (total_test_samples * num_classes)
        p_dominate = dominate_count / (total_test_samples * num_classes)

        # Log results for the current epoch
        log_msg = (f"Epoch: {epoch}, "
                   f"P_abnormal: {p_abnormal:.4%} ({abnormal_count}), "
                   f"P_dominate: {p_dominate:.4%} ({dominate_count})")
        print(log_msg)
        logger.log(log_msg)

        elapsed_time = time.time() - start_time
        print(f"Time: {elapsed_time:.2f}s")
        logger.log(f"Time: {elapsed_time:.2f}s")

        # Store the results
        p_abnormal_list.append(p_abnormal)
        p_dominate_list.append(p_dominate)

# --- Final Logging ---
# Log the complete lists for plotting or further analysis
logger.log(f"Final P_abnormal list: {p_abnormal_list}")
logger.log(f"Final P_dominate list: {p_dominate_list}")
print("Analysis complete. Results saved to:", log_path)
