import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

os.makedirs("docs", exist_ok=True)

print("Loading the perceptron and test data from 'models/' and 'data/' directory...")
weights_int: np.ndarray = np.load("models/weights.npy")
bias_int: np.ndarray = np.load("models/biases.npy")
X_test: np.ndarray = np.load("data/X_test.npy")
y_test: np.ndarray = np.load("data/y_test.npy")
history: Any = np.load("data/history.npz")

train_losses: np.ndarray = history["train_losses"]
test_losses: np.ndarray = history["test_losses"]
train_accs: np.ndarray = history["train_accs"]
test_accs: np.ndarray = history["test_accs"]

# Compute predictions for test evaluations
model_scores: np.ndarray = np.dot(X_test, weights_int.T) + bias_int
model_preds: np.ndarray = np.argmax(model_scores, axis=1)


# PLOT 1: Loss Curve (Train vs Test Loss)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", color="royalblue", linewidth=2)
plt.plot(test_losses, label="Test Loss (Validation)", color="darkorange", linestyle="--", linewidth=2)
plt.title("Loss Curve (Checks for Overfitting)", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("docs/01_loss_curve.png", dpi=300)
plt.close()
print("Saved: docs/01_loss_curve.png")


# PLOT 2: Generalization Gap (Accuracy Curve)

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train Accuracy", color="green", linewidth=2)
plt.plot(test_accs, label="Test Accuracy", color="crimson", linestyle="--", linewidth=2)
plt.title("Generalization Gap (Accuracy)", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("docs/02_accuracy_curve.png", dpi=300)
plt.close()
print("Saved: docs/02_accuracy_curve.png")


# PLOT 3: Confusion Matrix

cm: np.ndarray = confusion_matrix(y_test, model_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Quantized Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Digit")
plt.ylabel("True Digit")
plt.xticks(range(10))
plt.yticks(range(10))

for r in range(10):
    for c in range(10):
        text_color: str = "white" if cm[r, c] > (cm.max() / 2) else "black"
        plt.text(c, r, str(cm[r, c]), ha="center", va="center", color=text_color, fontsize=9)

plt.colorbar()
plt.tight_layout()
plt.savefig("docs/03_confusion_matrix.png", dpi=300)
plt.close()
print("Saved: docs/03_confusion_matrix.png")

print("SUCCESS: All 3 plots created successfully in 'docs/'!")