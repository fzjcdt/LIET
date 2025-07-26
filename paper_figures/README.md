## Running the Experiments

This code is used to reproduce the results for Figure 1, Figure 2, and Table 1 in the paper, which demonstrate the transferability of label information after catastrophic overfitting.

The process involves two main steps:

### Step 1: Train the Model and Save Checkpoints

First, run `co_test.py` to train the model. This script will save a model checkpoint for each training epoch, capturing the model's state as it undergoes catastrophic overfitting.

```bash
python co_test.py
```

### Step 2: Test Label Information Transferability

After the models from each epoch are saved, run `transfer_test.py`. This script will load the saved models and perform the tests to measure the transferability of label information, generating the data needed for the figures and table.

```bash
python transfer_test.py
```

### Configuration

You may need to adjust the following parameters directly in the code to match your experimental setup:

*   **`experiment_path`**: Before running, you may need to modify the `experiment_path` variable in the scripts to point to your desired directory for saving and loading models.
*   **Perturbation Size**: To experiment with different perturbation magnitudes (e.g., `epsilon`), modify the corresponding variable in the code.
*   **Base Inputs**: To test with different base inputs (e.g., uniform gray images, training data mean, uniform noise), you can adjust the relevant sections in the code where the input `x` is defined.