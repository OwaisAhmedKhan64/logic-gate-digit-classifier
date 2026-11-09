from typing import List
import numpy as np

weights: np.ndarray = np.load("models/weights.npy")
biases: np.ndarray = np.load("models/biases.npy")
X_test: np.ndarray = np.load("data/X_test.npy")
y_test: np.ndarray = np.load("data/y_test.npy")

w0: np.ndarray = weights[0]  # Shape: (64,)
b0: int = int(biases[0])
w0_grid: np.ndarray = w0.reshape(8, 8)

idx_0: int = int(np.where(y_test == 0)[0][0])
idx_1: int = int(np.where(y_test == 1)[0][0])

def analyze_sample(sample_pixels: np.ndarray, title: str, true_digit: int) -> None:
    grid: np.ndarray = sample_pixels.reshape(8, 8).astype(int)
    row_sums: List[int] = []

    print("\n" + "=" * 55)
    print(f" {title} (True Digit: {true_digit})")
    print("=" * 55)

    print("\n8x8 IMAGE (1 = ON, 0 = OFF):")
    print("     Col:  0 1 2 3 4 5 6 7")
    print("    -----------------------")
    for r in range(8):
        row_str: str = " ".join(str(val) for val in grid[r])
        visual: str = "".join(" ■" if val == 1 else " ." for val in grid[r])
        print(f"    Row {r}: [{row_str}]   {visual}")

    print("\n--- EXPECTED VALUES (ROW BY ROW) ---")
    total_pixel_sum: int = 0
    for r in range(8):
        row_px: np.ndarray = grid[r]
        row_w: np.ndarray = w0_grid[r]
        r_sum: int = int(np.sum(row_px * row_w))
        row_sums.append(r_sum)
        total_pixel_sum += r_sum
        print(f"  • Row {r} Multiplication Output: {r_sum:+3d}")

    final_score: int = total_pixel_sum + b0
    sign_bit: int = 1 if final_score < 0 else 0

    print("-" * 55)
    print(f"  Sum of all 8 Rows              : {total_pixel_sum:+d}")
    print(f"  Digit 0 Bias                   : {b0:+d}")
    print(f"  Expected Score (Score_0)       : {final_score:+d}")
    print(f"  Expected Sign Bit (Bit 7)      : {sign_bit}")
    print("=" * 55)

analyze_sample(X_test[idx_0], "TEST CASE 1: REAL DIGIT '0'", true_digit=0)
analyze_sample(X_test[idx_1], "TEST CASE 2: REAL DIGIT '1'", true_digit=1)