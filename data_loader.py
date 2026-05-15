import numpy as np
import pandas as pd
from pathlib import Path


class MNISTDataLoader:
    """Load, preprocess, and split the MNIST dataset."""

    def __init__(self, filepath, normalize=False, seed=None):
        self.filepath = Path(filepath)
        self.normalize = normalize
        self.seed = seed
        self._samples, self._labels = self._load()

    def _load(self):
        """Read the dataset from disk and return samples + labels"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found at '{self.filepath}'")
        
        df = pd.read_csv(self.filepath)
        labels = df["class"].to_numpy(dtype=np.int64)
        samples = df.drop(columns="class").to_numpy(dtype=np.float32)

        # Normalize pixel intensities for neural network training
        if self.normalize:
            samples /= 255.0
        
        return samples, labels
    
    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        return self._samples[idx], self._labels[idx]
    
    @property
    def samples(self):
        """Return the dataset samples"""
        return self._samples
    
    @property
    def labels(self):
        """Return the dataset labels"""
        return self._labels
    
    def train_test_split(self, test_size=0.3):
        """Split the dataset into randomized training and test splits"""

        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in the range (0, 1), got {test_size}")
        
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self))
        samples = self._samples[indices]
        labels = self._labels[indices]

        num_train = int(len(self) * (1 - test_size))

        x_train = samples[:num_train]
        x_test = samples[num_train:]

        y_train = labels[:num_train]
        y_test = labels[num_train:]
        
        return x_train, y_train, x_test, y_test

