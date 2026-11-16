from typing import List
import numpy as np

# Load test data and model
X_test: np.ndarray = np.load("data/X_test.npy")
y_test: np.ndarray = np.load("data/y_test.npy")
weights: np.ndarray = np.load("models/weights.npy")
biases: np.ndarray = np.load("models/biases.npy")

hex_64_list: List[str] = []

print("=" * 68)

for digit in range(10):
    idx: int = int(np.where(y_test == digit)[0][0])
    pixels: np.ndarray = X_test[idx].astype(int)
    grid: np.ndarray = pixels.reshape(8, 8)

    # Compute reference scores
    scores: np.ndarray = np.dot(pixels, weights.T) + biases
    predicted_digit: int = int(np.argmax(scores))

    # Convert to 64 bits hexadecimal 
    val_64: int = 0
    for bit_idx in range(64):
        if pixels[bit_idx] == 1:
            val_64 |= (1 << (63 - bit_idx))

    hex_str: str = f"{val_64:016x}"
    hex_64_list.append(hex_str)

    status: str = "MATCH" if digit == predicted_digit else "MISMATCH"

    print(f" Address {digit} (0x{digit:X})  |  True: {digit}  |  Predicted: {predicted_digit}  [{status}]")
    print(f" 64-bit Hex: {hex_str} \n")

# Write to hex file
with open("data/test_images_64.hex", "w") as f:
    f.write("v2.0 raw\n")
    f.write(" ".join(hex_64_list) + "\n")
    
print("=" * 68)
print("The file 'data/test_images_64.hex' has been created.")