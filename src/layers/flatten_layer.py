import numpy as np
from typing import Optional

class FlattenLayer:
    """
    Flatten layer for converting feature maps into a 1D feature vector.

    This layer reshapes a multi-dimensional tensor (C, H, W) into a
    single vector (C * H * W) so it can be fed into the dense layer.
    No trainable parameters, so this is purely for logistics purposes. 
    """

    def __init__(self):
        """
        Initialize flatten layer parameters.
        """

        # Store input shape for backward pass
        self.cache_input_shape: Optional[tuple] = None


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass of the flatten layer.

        :param x: Input tensor of shape (C, H, W)
        :type x: np.ndarray
        :return: Flattened feature vector
        :rtype: np.ndarray
        """

        self.cache_input_shape = x.shape

        # Flatten tensor into 1D vector
        output = x.reshape(-1)

        return output


    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the flatten layer.

        :param dL_dout: Gradient of the loss with respect to the flattened output
        :type dL_dout: np.ndarray
        :return: Gradient reshaped to original input dimensions
        :rtype: np.ndarray
        """

        # Reshape gradient back to original tensor shape
        dL_dinput = dL_dout.reshape(self.cache_input_shape)

        return dL_dinput