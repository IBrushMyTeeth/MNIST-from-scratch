import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def _reshape_image(
    sample: FloatArray,
    image_shape: tuple[int, int] =(28, 28)
) -> FloatArray:
    """Reshape a flattened image vector into image dimensions"""
    return sample.reshape(image_shape)

def show_image(
    sample: FloatArray,
    image_shape: tuple[int, int] = (28, 28),
    cmap: str ="gray"
) -> None:
    """Display a single image"""

    image = _reshape_image(sample, image_shape)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

def show_random_image(
    samples: FloatArray,
    labels: IntArray,
    rng: np.random.Generator | None = None,
    image_shape: tuple[int, int] =(28, 28),
    cmap: str ="gray"
) -> None:
    """Display a random image and its label"""

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, len(samples))
    row = samples[idx]
    label = labels[idx]

    image = _reshape_image(row, image_shape)

    plt.imshow(image, cmap=cmap)
    plt.title(f"Label: {label}", fontsize=16)
    plt.axis("off")
    plt.show()

def show_image_grid(
    samples: FloatArray,
    cols: int = 3,
    figsize_per_image: float = 2.0,
    image_shape: tuple[int, int] = (28, 28),
    cmap: str = "gray"
) -> None:
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


