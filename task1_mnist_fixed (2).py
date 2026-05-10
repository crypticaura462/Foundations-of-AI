"""
Task 1 – MNIST Dataset
(a) Downloads and loads the MNIST dataset
(b) Trains a classifier to distinguish digits 0–9

Requirements (install once):
    pip install scikit-learn matplotlib numpy

No TensorFlow or PyTorch needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# ══════════════════════════════════════════════════════════════════
#  TASK 1(a) – Download & Load MNIST
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TASK 1(a) – Downloading MNIST Dataset")
print("=" * 60)

# fetch_openml downloads MNIST automatically and caches it locally.
# First run takes ~1–2 minutes; subsequent runs are instant.
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")

X = mnist.data        # shape (70000, 784) – each row is a 28×28 image flattened
y = mnist.target.astype(int)  # labels 0–9

print(f"\n  Total samples  : {X.shape[0]:,}")
print(f"  Features/image : {X.shape[1]}  (28 × 28 pixels)")
print(f"  Classes        : {sorted(set(y.tolist()))}")

# ── Visualise a few sample digits ─────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
fig.suptitle("Sample MNIST digits (0 – 9)", fontsize=12)
for digit in range(10):
    idx = np.where(y == digit)[0][0]          # first occurrence of each digit
    for row in range(2):
        idx2 = np.where(y == digit)[0][row]
        axes[row, digit].imshow(X[idx2].reshape(28, 28), cmap="gray")
        axes[row, digit].set_title(str(digit), fontsize=9)
        axes[row, digit].axis("off")
plt.tight_layout()
plt.savefig("mnist_samples.png", dpi=100)
plt.show()
print("\n  Sample image grid saved → mnist_samples.png")

# ══════════════════════════════════════════════════════════════════
#  TASK 1(b) – Build & Train Digit Classifier
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 1(b) – Training Digit Classifier (0–9)")
print("=" * 60)

# ── 1. Split ──────────────────────────────────────────────────────
# Use the standard MNIST split: 60k train / 10k test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f"\n  Training set : {len(X_train):,} samples")
print(f"  Test set     : {len(X_test):,} samples")

# ── 2. Normalise ──────────────────────────────────────────────────
# Scale pixel values from [0, 255] → mean≈0, std≈1
# (helps the neural network learn faster and more stably)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)       # use SAME scaler, not refit

# ── 3. Define the model ───────────────────────────────────────────
# MLPClassifier = Multi-Layer Perceptron (feedforward neural network)
#   hidden_layer_sizes: two hidden layers with 256 and 128 neurons
#   activation        : ReLU activation function
#   solver            : Adam optimiser
#   max_iter          : maximum training epochs
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=20,
    random_state=42,
    verbose=True,        # prints loss each epoch
)

print("\n  Model architecture:")
print("    Input  : 784 neurons  (28×28 pixels)")
print("    Hidden : 256 neurons  (ReLU)")
print("    Hidden : 128 neurons  (ReLU)")
print("    Output :  10 neurons  (digits 0–9)")
print("\n  Training …\n")

model.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION ON TEST SET")
print("=" * 60)

y_pred    = model.predict(X_test)
test_acc  = accuracy_score(y_test, y_pred)

print(f"\n  Overall Test Accuracy : {test_acc * 100:.2f}%")
print("\n  Per-class Report:\n")
print(classification_report(y_test, y_pred,
      target_names=[f"Digit {d}" for d in range(10)]))

# ── 5. Per-digit bar chart ────────────────────────────────────────
per_digit_acc = []
for digit in range(10):
    mask = y_test == digit
    acc  = accuracy_score(y_test[mask], y_pred[mask])
    per_digit_acc.append(acc)
    bar  = "█" * int(acc * 25)
    print(f"  Digit {digit}: {acc*100:5.1f}%  {bar}")

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(range(10), [a * 100 for a in per_digit_acc],
              color=plt.cm.tab10.colors, edgecolor="white", linewidth=.5)
ax.set_xticks(range(10))
ax.set_xticklabels([f"Digit {d}" for d in range(10)], rotation=30)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-digit Classification Accuracy on MNIST Test Set")
ax.set_ylim(90, 100)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig("mnist_per_digit_accuracy.png", dpi=100)
plt.show()
print("\n  Bar chart saved → mnist_per_digit_accuracy.png")

# ── 6. Confusion matrix ───────────────────────────────────────────
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=list(range(10)))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Confusion Matrix – MNIST Test Set")
plt.tight_layout()
plt.savefig("mnist_confusion_matrix.png", dpi=100)
plt.show()
print("  Confusion matrix saved → mnist_confusion_matrix.png")

# ── 7. Show a few predictions ─────────────────────────────────────
wrong_idx = np.where(y_pred != y_test)[0][:10]   # first 10 wrong ones
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
fig.suptitle("Top row: correct predictions  |  Bottom row: mis-classifications",
             fontsize=10)

correct_idx = np.where(y_pred == y_test)[0][:10]
for i in range(10):
    # correct
    img = scaler.inverse_transform([X_test[correct_idx[i]]])[0].reshape(28, 28)
    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].set_title(f"✓ {y_pred[correct_idx[i]]}", fontsize=8, color="green")
    axes[0, i].axis("off")
    # wrong
    img2 = scaler.inverse_transform([X_test[wrong_idx[i]]])[0].reshape(28, 28)
    axes[1, i].imshow(img2, cmap="gray")
    axes[1, i].set_title(f"✗ pred={y_pred[wrong_idx[i]]}\ntrue={y_test[wrong_idx[i]]}",
                         fontsize=7, color="red")
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig("mnist_predictions.png", dpi=100)
plt.show()
print("  Prediction samples saved → mnist_predictions.png")

print("\n  All done!")
