import numpy as np
from typing import Tuple, Optional


class MaxPoolLayer:
    """
    Max pooling layer for spatial downsampling of feature maps.

    Selects the maximum value inside each pooling window.
    """

    def __init__(self, pool_size: int, stride: int):
        """
        Initialize pooling parameters.

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

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute forward pass of the max pooling layer.

        :param input_data: Input tensor of shape (C, H, W)
        :type input_data: np.ndarray
        :return: Downsampled feature maps
        :rtype: np.ndarray
        """

        self.cache_input = input_data  # store input for backward pass
        self.num_channels, self.input_height, self.input_width = input_data.shape

        # Compute output spatial dimensions
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        # Loop over channels and output positions
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):

                    # Calculate window boundaries
                    start_i = i * self.pool_size
                    end_i = start_i + self.pool_size
                    start_j = j * self.pool_size
                    end_j = start_j + self.pool_size

                    # Extract the patch from input
                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    # Assign the maximum value in the patch to output
                    self.output[c, i, j] = np.max(patch)

        return self.output
    
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Compute backward pass of max pooling layer.

        :param dL_dout: Gradient of the loss w.r.t the layer output (C, H_out, W_out)
        :type dL_dout: np.ndarray
        :return: Gradient of the loss w.r.t the input (C, H, W)
        :rtype: np.ndarray
        """

        # Initialize gradient array w.r.t input
        dL_dinput = np.zeros_like(self.cache_input)
        
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):

                    start_i = i * self.pool_size
                    end_i = start_i + self.pool_size
                    start_j = j * self.pool_size
                    end_j = start_j + self.pool_size

                    patch = self.cache_input[c, start_i:end_i, start_j:end_j]

                    # Find max value index in the patch
                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)

                    # Route gradient from output to max element
                    dL_dinput[c,
                            start_i + max_idx[0],
                            start_j + max_idx[1]] += dL_dout[c, i, j]

        return dL_dinput