import numpy as np

print("Loading the perceptron and test data from 'models/' and 'data/'.")
weights: np.ndarray = np.load("models/weights.npy")
biases: np.ndarray = np.load("models/biases.npy")
X_test: np.ndarray = np.load("data/X_test.npy")
y_test: np.ndarray = np.load("data/y_test.npy")

# Do the math: Score = Sum(X * W) + Bias
scores: np.ndarray = np.dot(X_test, weights.T) + biases
predictions: np.ndarray = np.argmax(scores, axis=1)

# Calculate accuracy
accuracy: float = float(np.mean(predictions == y_test) * 100.0)

print(f"Accuracy: {accuracy:.2f}%")

# Generate a sample test case
sample_idx: int = 0
sample_pixels: np.ndarray = X_test[sample_idx]
sample_true: int = int(y_test[sample_idx])
sample_pred: int = int(predictions[sample_idx])
sample_scores: np.ndarray = scores[sample_idx]

# Show sample test case
print(f"Sample Test Case")
print(f"True Digit     : {sample_true}")
print(f"Predicted Digit: {sample_pred}")
print(f"Digit Scores   : {sample_scores}")
