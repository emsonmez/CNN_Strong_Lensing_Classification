import numpy as np
from typing import Optional


class DenseLayer:
    """
    Fully (dense) connected layer for classification.

    Combines extracted CNN features and maps them to class probabilities.
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

        self.cache_input = x

        # Linear transformation
        z = np.dot(self.weight, x) + self.bias

        # Softmax non-linear activation function
        z_shifted = z - np.max(z) # Shift input values for numerical stability 
        exp_z = np.exp(z_shifted)
        self.output = exp_z / np.sum(exp_z)

        return self.output
    
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

        # Initialize gradients
        dL_dinput = np.zeros_like(x) # gradient w.r.t input
        dL_dweight = np.zeros_like(self.weight) # gradient w.r.t weights
        dL_dbias = np.zeros_like(self.bias) # gradient w.r.t bias

        # Gradient of loss w.r.t linear output
        dL_dz = dL_dout

        # Loop over output neurons
        for i in range(self.output_size):

            # Compute gradient of loss w.r.t weights
            dL_dweight[i] = dL_dz[i] * x

            # Compute gradient of loss w.r.t input
            dL_dinput += dL_dz[i] * self.weight[i]

            # Compute gradient of loss w.r.t bias
            dL_dbias[i] = dL_dz[i]

        # Store gradients for optimizer
        self.dL_dweight = dL_dweight
        self.dL_dbias = dL_dbias

        # Update layer parameters
        self.weight -= lr * dL_dweight
        self.bias -= lr * dL_dbias

        return dL_dinput