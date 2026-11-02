import os
import random
from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Fixing the seed to produce same results
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# Load the dataset
digits: Any = load_digits()

# Binarize the image pixels with a threshold
binarizing_threshold: int = 6
X: np.ndarray = (digits.data > binarizing_threshold).astype(np.float32)
y: np.ndarray = digits.target

# Splitting the dataset
X_train: np.ndarray
X_test: np.ndarray
y_train: np.ndarray
y_test: np.ndarray
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Turning numpy arrays to tensors
X_train_tensor: torch.Tensor = torch.tensor(X_train)
y_train_tensor: torch.Tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor: torch.Tensor = torch.tensor(X_test)
y_test_tensor: torch.Tensor = torch.tensor(y_test, dtype=torch.long)

# Defining the perceptron (neural network) model
class HardwarePerceptron(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# Instantiating the model
model: HardwarePerceptron = HardwarePerceptron()

# Defining the loss function and optimizer
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)

# Tracking history for plotting
train_losses: List[float] = []
test_losses: List[float] = []
train_accs: List[float] = []
test_accs: List[float] = []

# Training the model
print("Training the perceptron.")
epoch_size: int = 200
for epoch in range(epoch_size):
    optimizer.zero_grad()
    train_out: torch.Tensor = model(X_train_tensor)
    loss: torch.Tensor = criterion(train_out, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Track metrics per epoch
    with torch.no_grad():
        test_out: torch.Tensor = model(X_test_tensor)
        t_loss: torch.Tensor = criterion(test_out, y_test_tensor)

        train_pred: torch.Tensor = torch.argmax(train_out, dim=1)
        test_pred: torch.Tensor = torch.argmax(test_out, dim=1)

        t_acc: float = float((train_pred == y_train_tensor).float().mean().item() * 100.0)
        val_acc: float = float((test_pred == y_test_tensor).float().mean().item() * 100.0)

        train_losses.append(float(loss.item()))
        test_losses.append(float(t_loss.item()))
        train_accs.append(t_acc)
        test_accs.append(val_acc)

# Extracting weights and biases
weights_float: np.ndarray = model.fc.weight.data.numpy()
bias_float: np.ndarray = model.fc.bias.data.numpy()

# Scaling the weights and biases to range of -4 to 4
scale: float = float(4.0 / np.percentile(np.abs(weights_float), 99))
weights_int: np.ndarray = np.clip(np.round(weights_float * scale).astype(int), -4, 4)
bias_int: np.ndarray = np.round(bias_float * scale).astype(int)

# Saving the model artifacts
np.save("models/weights.npy", weights_int)
np.save("models/biases.npy", bias_int)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

# Saving training history
np.savez(
    "data/history.npz",
    train_losses=np.array(train_losses, dtype=np.float32),
    test_losses=np.array(test_losses, dtype=np.float32),
    train_accs=np.array(train_accs, dtype=np.float32),
    test_accs=np.array(test_accs, dtype=np.float32),
)

# Creating a blueprint for the weights
with open("docs/weights_blueprint.txt", "w") as f:
    f.write("=== PERCEPTRON WEIGHT BLUEPRINT (Range: -4 to +4) ===\n\n")
    for digit in range(10):
        f.write(f"--- DIGIT {digit} (Bias = {bias_int[digit]}) ---\n")
        grid: np.ndarray = weights_int[digit].reshape(8, 8)
        for row in grid:
            f.write("  " + "  ".join(f"{val:2d}" for val in row) + "\n")
        f.write("\n")

print("Training complete.")