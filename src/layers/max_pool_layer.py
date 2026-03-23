from typing import Optional
import numpy as np


class MaxPoolLayer:
    """
    Max pooling layer for spatial downsampling of feature maps.

    Selects the maximum value inside each pooling window.
    """

    def __init__(self, pool_size: int, stride: int):
        """
        Initialize max pooling parameters.

        :param pool_size: Height and width of pooling window
        :type pool_size: int
        :param stride: Step size of pooling window movement
        :type stride: int
        """
        self.pool_size = pool_size
        self.stride = stride

        # Stored for backward pass
        self.cache_input: Optional[np.ndarray] = None

        # Input dimensions
        self.num_channels: Optional[int] = None
        self.input_height: Optional[int] = None
        self.input_width: Optional[int] = None

        # Output dimensions
        self.output_height: Optional[int] = None
        self.output_width: Optional[int] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the max pooling layer.

        :param x: Input tensor of shape (C, H, W)
        :type x: np.ndarray
        :return: Downsampled feature maps
        :rtype: np.ndarray
        """

        single_image = False
        if x.ndim == 3:  # Add batch dimension for single image
            x = x[np.newaxis, :, :, :]
            single_image = True
        self.cache_input = x  # store input for backward pass
        self.single_image = single_image

        self.batch_size, self.num_channels, self.input_height, self.input_width = (
            x.shape
        )

        # Compute output spatial dimensions
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output = np.zeros(
            (self.batch_size, self.num_channels, self.output_height, self.output_width)
        )

        # Loop over channels and output positions
        for n in range(self.batch_size):
            for c in range(self.num_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        # Calculate window boundaries
                        start_i = i * self.pool_size
                        end_i = start_i + self.pool_size
                        start_j = j * self.pool_size
                        end_j = start_j + self.pool_size

                        # Extract the patch from input
                        patch = x[n, c, start_i:end_i, start_j:end_j]

                        # Assign the maximum value in the patch to output
                        self.output[n, c, i, j] = np.max(patch)

        return self.output[0] if self.single_image else self.output

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the max pooling layer.

        :param dL_dout: Gradient of the loss w.r.t the layer output (C, H_out, W_out)
        :type dL_dout: np.ndarray
        :return: Gradient of the loss w.r.t the input (C, H, W)
        :rtype: np.ndarray
        """

        x = self.cache_input

        # Ensure dL_dout is also 4D
        if self.single_image:
            dL_dout = dL_dout[np.newaxis, :, :, :]

        self.batch_size, self.num_channels, self.input_height, self.input_width = (
            x.shape
        )

        # Initialize gradient array w.r.t input
        dL_dinput = np.zeros_like(self.cache_input)

        # Loop over channels and output positions
        for n in range(self.batch_size):
            for c in range(self.num_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.pool_size
                        end_i = start_i + self.pool_size
                        start_j = j * self.pool_size
                        end_j = start_j + self.pool_size

                        patch = self.cache_input[n, c, start_i:end_i, start_j:end_j]

                        # Find max value index in the patch
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)

                        # Route gradient from output to max element
                        dL_dinput[
                            n, c, start_i + max_idx[0], start_j + max_idx[1]
                        ] += dL_dout[n, c, i, j]

        return dL_dinput[0] if self.single_image else dL_dinput
