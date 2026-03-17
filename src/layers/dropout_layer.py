import numpy as np 
from typing import Optional

class DropoutLayer:
    """
    Dropout layer for regularization.

    Randomly sets a fraction of input units to zero during training
    and scales the remaining activations to maintain expected value.
    """

    def __init__(self, dropout_rate: float = 0.2):
        """
        Initialize dropout layer parameters.

        :param dropout_rate: Probability of dropping a neuron
        :type dropout_rate: float
        """

        self.dropout_rate = dropout_rate

        self.mask: Optional[np.ndarray] = None


    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute the forward pass of the dropout layer.

        :param x: Input tensor
        :type x: np.ndarray
        :param training: Whether the model is in training mode
        :return training: bool
        :return: Output tensor after applying dropout
        :rtype: np.ndarray
        """

        if not training:
            return x

        # Dropout mask (1 = keep, 0 = drop)
        self.mask = (np.random.rand(*x.shape) > self.dropout_rate)

        # Inverted dropout scaling
        output = (x * self.mask) / (1.0 - self.dropout_rate)

        return output
    
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the dropout layer.

        :param dL_dout: Gradient of the loss with respect to the layer output
        :type dL_dout: np.ndarray
        :return: Gradient of the loss with respect to the input
        :rtype: np.ndarray
        """

        if self.mask is None:
            return dL_dout

        # Backprop through mask with same scaling
        dL_dinput = (dL_dout * self.mask) / (1.0 - self.dropout_rate)

        return dL_dinput

