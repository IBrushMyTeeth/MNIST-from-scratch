# MNIST From Scratch

A complete deep learning pipeline built **from scratch using only NumPy** — from raw data loading and preprocessing, through a configurable multi-layer perceptron with manual backpropagation, to model evaluation and prediction inspection.

This project was created as a learning exercise to understand every stage of a neural network pipeline without relying on deep learning frameworks such as TensorFlow or PyTorch.

---

## Pipeline Overview

```
Raw Data → Preprocessing → Model Definition → Training → Evaluation → Serialization → Inspection
```

Each stage is implemented manually:

1. **Data** — Download, load, and preprocess the MNIST dataset
2. **Preprocessing** — Normalize pixel values, split into train/test sets, generate mini-batches
3. **Model** — Build a configurable MLP with custom layer sizes and activation functions
4. **Training** — Forward propagation, backpropagation, and mini-batch gradient descent
5. **Evaluation** — Accuracy reporting and per-class breakdown
6. **Serialization** — Save and reload trained model weights
7. **Inspection** — Visualize predictions on random samples

---

## Project Structure

```
.
├── data/
│   └── mnist_2000.csv
│
├── models/
│   └── mnist_model.npz
│
├── download.py
├── data_loader.py
├── data_visualization.py
├── network.py
├── training.py
├── inspect_predictions.py
└── README.md
```

---

## Quickstart

### 1. Download the Dataset

```bash
python download.py
```

This fetches MNIST from OpenML via `scikit-learn` and stores the first 2000 samples to `data/mnist_2000.csv`.

### 2. Train the Model

```bash
python training.py
```

### 3. Inspect Predictions

```bash
python inspect_predictions.py
```

---

## Stage-by-Stage Breakdown

### Stage 1 — Data Download

The dataset is fetched from OpenML using `scikit-learn` and saved locally as a CSV:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=True)

rows = 2000
df = mnist.frame[:rows]
df.to_csv(f"./data/mnist_{rows}.csv", index=False)
```

---

### Stage 2 — Preprocessing

The data loader handles:

- Loading the CSV into NumPy arrays
- Optional pixel normalization to `[0, 1]`
- Random train/test splitting
- Mini-batch generation for gradient descent

---

### Stage 3 — Model Definition

The MLP is fully configurable by specifying layer sizes and activation functions:

```python
layers = [784, 128, 32, 10]
activations = ["relu", "relu", "softmax"]

model = MLP(layers, activations)
```

The implementation validates that:

- Input size is 784 (28×28 pixels)
- Output size is 10 (one per digit class)
- The number of activations matches the number of layers
- The final activation is Softmax

**Supported activations:** ReLU, Tanh, Softmax

**Supported weight initialization:** He (for ReLU layers), Xavier (for Tanh layers)

---

### Stage 4 — Training

The training loop implements the full learning pipeline from scratch:

- **Forward propagation** — Compute activations layer by layer
- **Cross-entropy loss** — Measure prediction error
- **Backpropagation** — Compute gradients with respect to every weight and bias
- **Gradient descent** — Update parameters using mini-batches

```bash
python training.py
```

Default architecture: `784 → 128 → 32 → 10`

---

### Stage 5 — Evaluation

After training, the model reports:

- Total samples tested
- Number correct and incorrect
- Overall accuracy
- Per-class accuracy for each digit (0–9)

```
========================================
MNIST Evaluation Report
========================================

Samples Tested : 600
Correct        : 564
Incorrect      : 36

Accuracy       : 94.00%

Per-Class Accuracy

Digit 0: Accuracy = ...
Digit 1: Accuracy = ...
...
Digit 9: Accuracy = ...
```

---

### Stage 6 — Serialization

Trained model weights can be saved to disk and restored later:

```python
# Save
model.save("models/mnist_model.npz")

# Restore
model.set_parameters(weights, biases)
```

---

### Stage 7 — Prediction Inspection

```bash
python inspect_predictions.py
```

Loads the saved model, selects random test images, and displays them in a grid alongside their predicted and actual labels — a simple visual sanity check of the full pipeline.

---

## Educational Goal

The primary goal is to understand what actually happens inside a neural network by building every component by hand.

Rather than calling `model.fit()`, this project implements each stage of the pipeline explicitly — from how raw pixel values become training batches, to how gradients flow backwards through each layer, to how a trained model gets written to and read from disk.

Every stage that a framework would normally hide is here, readable, and broken down into its constituent parts.