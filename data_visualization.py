import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def _reshape_image(sample, image_shape=(28, 28)):
    """Reshape a flattened image vector into image dimensions"""
    return sample.reshape(image_shape)

def show_image(sample, image_shape=(28, 28), cmap="gray"):
    """Display a single image"""
    image = _reshape_image(sample, image_shape)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

def show_random_image(samples, labels, rng, image_shape=(28, 28), cmap="gray"):
    """Display a random image and its label"""
    idx = rng.integers(0, len(samples))
    row = samples[idx]
    label = labels[idx]

    image = _reshape_image(row, image_shape)

    plt.imshow(image, cmap=cmap)
    plt.title(f"Label: {label}", fontsize=16)
    plt.axis("off")
    plt.show()

def show_image_grid(samples,
    cols=3,
    figsize_per_image=2,
    image_shape=(28, 28),
    cmap="gray"):

    """Display multiple images in a grid layout"""

    n = len(samples)
    rows = ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_image,
                 rows * figsize_per_image)
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, sample in zip(axes, samples):
        ax.imshow(_reshape_image(sample, image_shape), cmap=cmap)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


