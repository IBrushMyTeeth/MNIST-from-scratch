import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class MLP:
    """
    This class represents a validated MNIST-compatible feedforward classifier
    with fixed input/output semantics.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        activations: list[str] | None = None,
        seed: int | None = None
    ) -> None:

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.activations_map = {
            "relu": self._relu,
            "tanh": self._tanh,
            "softmax": self._softmax,
        }

        self.activation_derivatives = {
            "relu": self._relu_derivative,
            "tanh": self._tanh_derivative,
        }

        self.layers = layers or [784, 128, 32, 10]
        self.activations = self._resolve_activations(activations)

        self._validate_architecture()

        self._build_network()

        self.z_values: list[FloatArray] = []
        self.a_values: list[FloatArray] = []

    def _resolve_activations(
        self,
        activations: list[str] | None
    ) -> list[str]:
        """
        Create default activations if none are supplied.
        """

        if activations is None:
            return (
                ["relu"] * (len(self.layers) - 2)
                + ["softmax"]
            )

        return activations
        
    def _build_network(self) -> None:
        """
        Construct trainable parameters.
        """

        self.weight_matrices: list[FloatArray] = []
        self.bias_vectors: list[FloatArray] = []

        for in_dim, out_dim, activation in zip(
            self.layers[:-1],
            self.layers[1:],
            self.activations
        ):

            if activation == "relu":
                weight = self._he_initialization(in_dim, out_dim)

            else:
                weight = self._xavier_initialization(in_dim, out_dim)

            self.weight_matrices.append(weight)
            self.bias_vectors.append( np.zeros(out_dim, dtype=np.float32))
        

    def _validate_architecture(self) -> None:
        """
        Validate user-provided network architecture.
        """

        if len(self.layers) < 2:
            raise ValueError(
                "A neural network requires at least an input and output layer."
            )

        if self.layers[0] != 784:
            raise ValueError(
                f"Expected input size 784, got {self.layers[0]}."
            )

        if self.layers[-1] != 10:
            raise ValueError(
                f"Expected output size 10, got {self.layers[-1]}."
            )

        if any(size <= 0 for size in self.layers):
            raise ValueError(
                "All layer sizes must be positive integers."
            )

        expected_activations = len(self.layers) - 1

        if len(self.activations) != expected_activations:
            raise ValueError(
                f"Expected {expected_activations} activations, "
                f"got {len(self.activations)}."
            )

        for activation in self.activations:
            if activation not in self.activations_map:
                raise ValueError(
                    f"Unknown activation '{activation}'."
                )

        if self.activations[-1] != "softmax":
            raise ValueError(
                "Output layer must use softmax for MNIST classification."
            )        
    
    def _clear_cache(self):
        """Remove cached forward-pass values"""
        self.z_values = []
        self.a_values = []
        
    def _relu(self, x: FloatArray) -> FloatArray:
        return np.maximum(x, 0)

    def _relu_derivative(self, z: FloatArray) -> FloatArray:
        return (z > 0).astype(np.float32)

    def _tanh(self, x: FloatArray) -> FloatArray:
        return np.tanh(x)

    def _tanh_derivative(self, z: FloatArray) -> FloatArray:
        return (1 - np.tanh(z) ** 2)
    
    def _softmax(self, x: FloatArray) -> FloatArray:
        # Shift values for numerical stability
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        # Convert logits into class probabilities
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _he_initialization(self, in_dim: int, out_dim: int)-> FloatArray:
        weights = self.rng.standard_normal((in_dim, out_dim), dtype=np.float32)
        return weights * np.sqrt(2/in_dim)
    
    def _xavier_initialization(self, in_dim: int, out_dim: int) -> FloatArray:
        weights = self.rng.standard_normal((in_dim, out_dim), dtype=np.float32)
        return weights * np.sqrt(1/in_dim) 

    def forward(self, x: FloatArray, training: bool = False) -> FloatArray:
        """Run a forward pass through the network"""
        if x.ndim != 2:
            raise ValueError(
                f"Expected shape (batch_size, {self.layers[0]}), got {x.shape}")
        
        if training:
            self._clear_cache()
            self.a_values.append(x)

        for weight,bias, activation in zip(
                self.weight_matrices,
                self.bias_vectors,
                self.activations
        ):
            
            z = (x @ weight + bias)
            a = self.activations_map[activation](z)

            if training:
                self.z_values.append(z)
                self.a_values.append(a)

            x = a

        return x

    def predict(self, x: FloatArray) -> IntArray:
        """Predict a single label for each sample"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.argmax(self.forward(x), axis=1)
    
    def parameters(self):
        """Return each weight and its corresponding bias"""
        return zip(self.weight_matrices, self.bias_vectors)
    
    def update_parameters(
            self,
            grad_w: list[FloatArray],
            grad_b: list[FloatArray],
            learning_rate: float = 0.01
    ) -> None:
        """Update model parameters based on learning rate and gradients"""
        for i in range(len(self.weight_matrices)):
            self.weight_matrices[i] -= learning_rate * grad_w[i]
            self.bias_vectors[i] -= learning_rate * grad_b[i]