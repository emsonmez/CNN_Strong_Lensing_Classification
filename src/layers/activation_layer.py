from typing import Optional
import numpy as np


class ActivationLayer:
    """
    Leaky ReLU activation layer.

    Applies a non-linear activation function that allows a small,
    non-zero gradient when the unit is not active.
    """

    def __init__(self, alpha: float = 0.01):
        """
        Initialize Leaky ReLU activation layer parameters.

        :param alpha: Slope for negative input values
        :type alpha: float
        """
        self.alpha = alpha

        # Stored for backward pass
        self.cache_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the Leaky ReLU activation layer.

        :param x: Input tensor
        :type x: np.ndarray
        :return: Activated output tensor
        :rtype: np.ndarray
        """

        self.cache_input = x

        # Initialize output tensor
        output = np.zeros_like(x)

        # Apply Leaky ReLU element-wise
        for idx, value in np.ndenumerate(x):
            if value >= 0:
                output[idx] = value
            else:
                output[idx] = self.alpha * value

        return output

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the Leaky ReLU activation layer.

        :param dL_dout: Gradient of the loss with respect to the layer output
        :type dL_dout: np.ndarray
        :return: Gradient of the loss with respect to the input
        :rtype: np.ndarray
        """

        x = self.cache_input

        # Initialize gradient w.r.t input
        dL_dinput = np.zeros_like(x)

        # Compute gradient element-wise
        for idx, value in np.ndenumerate(x):
            if value >= 0:
                dL_dinput[idx] = dL_dout[idx]
            else:
                dL_dinput[idx] = dL_dout[idx] * self.alpha

        return dL_dinput
