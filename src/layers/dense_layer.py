from typing import Optional
import numpy as np


class DenseLayer:
    """
    Fully (dense) connected layer for classification.

    Combines extracted CNN features and maps them to class probabilities.
    Follows the softmax formulation. Generalized for both single-image
    and batch inputs.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize dense layer parameters.

        :param input_size: Number of input features
        :type input_size: int
        :param output_size: Number of output neurons (classes)
        :type output_size: int
        """

        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights and bias
        self.weight = np.random.randn(output_size, input_size)
        self.bias = np.zeros(output_size)

        # Stored for backward pass
        self.cache_input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the dense layer.

        :param x: Input vector
        :type x: np.ndarray
        :return: Output probabilities after softmax activation
        :rtype: np.ndarray
        """

        single_input = False
        # Ensure batch dimension
        if x.ndim == 1:  # (input_size,)
            x = x[np.newaxis, :]
            single_input = True

        self.cache_input = x
        self.single_input = single_input

        self.batch_size, self.input_size = x.shape

        # Linear transformation
        z = x @ self.weight.T + self.bias  # (N, output_size)

        # Softmax non-linear activation function (numerically stable)
        z_shifted = z - np.max(
            z, axis=1, keepdims=True
        )  # Shift input values for numerical stability
        exp_z = np.exp(z_shifted)
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return self.output[0] if self.single_input else self.output

    def backward(self, dL_dout: np.ndarray, lr: float) -> np.ndarray:
        """
        Compute the backward pass of the dense layer.

        :param dL_dout: Gradient of the loss with respect to the layer output
        :type dL_dout: np.ndarray
        :param lr: Learning rate for parameter updates
        :type lr: float
        :return: Gradient of the loss with respect to the input
        :rtype: np.ndarray
        """

        x = self.cache_input

        if self.single_input:
            dL_dout = dL_dout[np.newaxis, :]

        # Batch size
        self.batch_size = x.shape[0]

        # Initialize gradients
        dL_dweight = dL_dout.T @ x  # (output_size, input_size)
        dL_dbias = np.sum(dL_dout, axis=0)  # (output_size,)
        dL_dinput = dL_dout @ self.weight  # (N, input_size)

        # Average over batch
        dL_dweight /= self.batch_size
        dL_dbias /= self.batch_size

        # Store gradients for optimizer
        self.dL_dweight = dL_dweight
        self.dL_dbias = dL_dbias

        # Update layer parameters (only if lr > 0)
        self.weight -= lr * dL_dweight
        self.bias -= lr * dL_dbias

        return dL_dinput[0] if self.single_input else dL_dinput
