import numpy as np
from network import MLP
from data_loader import MNISTDataLoader
from data_visualization import *

architecture = np.load("models/mnist_model.npz", allow_pickle=True)

layers = architecture["layers"].tolist()
activations = architecture["activations"].tolist()
weights = architecture["weights"].tolist()
biases = architecture["biases"].tolist()


model = MLP(layers, activations)
model.set_parameters(weights, biases)


loader = MNISTDataLoader("data/mnist_2000.csv", normalize=True, seed=21)


################### INSPECTION ###################
seed = 42
rng = np.random.default_rng(seed)
num_samples = 25
rounds = 2
x = loader.samples
y = loader.labels

for round in range(rounds):
    idx = rng.choice(len(loader), num_samples, replace=False)
    samples = x[idx]
    labels = y[idx]
    pred = model.predict(samples)


    print("=" * 10 + f" Predictions at round {round + 1} " + "=" * 10)
    for i in range(num_samples):
        print(f"Predicted label: {pred[i]} " + " "* 10 + f"True label: {labels[i]}")
    
    accuracy = np.mean(pred == labels)
    print(f"Batch accuracy: {accuracy:.2%}")
    print()
    
    show_image_grid(samples, cols= 5)