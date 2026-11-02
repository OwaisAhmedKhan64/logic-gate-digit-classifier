import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
fig, axes = plt.subplots(2, 10, figsize=(14, 4))
binarizing_threshold: int = 6

for target_digit in range(10):
    idx = (digits.target == target_digit).argmax()
    original_img = digits.images[idx]
    binarized_img = (original_img > binarizing_threshold).astype(int)

    axes[0, target_digit].imshow(original_img, cmap="gray")
    axes[0, target_digit].set_title(f"Label: {target_digit}")
    axes[0, target_digit].axis("off")

    axes[1, target_digit].imshow(binarized_img, cmap="binary")
    axes[1, target_digit].set_title(f"Binarized {target_digit}")
    axes[1, target_digit].axis("off")
    
axes[0, 0].set_ylabel("Original Grayscale", size=12, labelpad=30)
axes[1, 0].set_ylabel("Inputs (0/1)", size=12, labelpad=30)

plt.suptitle(f"8x8 Handwritten Digits (Original vs Binary Input) with Threshold = {binarizing_threshold}", fontsize=14)
plt.tight_layout()

# Saving the image in the docs/ directory
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/binarized_pixels_data_comparison.png", dpi=200)
print("Saved image to docs/binarized_pixels_data_comparison.png")

plt.show()