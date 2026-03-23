from typing import Optional
import numpy as np


class FlattenLayer:
    """
    Flatten layer for converting feature maps into a 1D feature vector.

    This layer reshapes a multi-dimensional tensor (C, H, W) into a
    single vector (C * H * W) so it can be fed into the dense layer.
    No trainable parameters, so this is purely for logistics purposes.
    Generalized for both single-image and batch inputs.
    """

    def __init__(self):
        """
        Initialize flatten layer parameters.
        """

        # Store input shape for backward pass
        self.cache_input_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the flatten layer.

        :param x: Input tensor of shape (C, H, W)
        :type x: np.ndarray
        :return: Flattened feature vector
        :rtype: np.ndarray
        """

        self.single_image = False
        if x.ndim == 3:
            x = x[np.newaxis, ...]
            self.single_image = True

        self.cache_input_shape = x.shape

        # Flatten all dimensions except batch
        output = x.reshape(x.shape[0], -1)

        return output[0] if self.single_image else output

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the flatten layer.

        :param dL_dout: Gradient of the loss with respect to the flattened output
        :type dL_dout: np.ndarray
        :return: Gradient reshaped to original input dimensions
        :rtype: np.ndarray
        """

        if self.single_image:
            dL_dout = dL_dout[np.newaxis, ...]

        # Reshape gradient back to original tensor shape
        dL_dinput = dL_dout.reshape(self.cache_input_shape)

        return dL_dinput[0] if self.single_image else dL_dinput
