import numpy as np
from scipy.signal import correlate2d
from typing import Optional, Tuple


class ConvLayer:
    """
    A single 2D convolutional layer.

    Applies learnable filters across spatial regions of the input tensor.
    Produces one feature map per filter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0
    ):
        """
        Initialize convolution layer parameters.

        :param in_channels: Number of input channels
        :type in_channels: int
        :param out_channels: Number of convolution filters
        :type out_channels: int
        :param kernel_size: Kernel height and width
        :type kernel_size: Tuple[int, int]
        :param stride: Step size of kernel movement
        :type stride: int
        :param padding: Zero-padding applied to input borders
        :type padding: int
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k_h, k_w = kernel_size

        # He-style initialization for weights
        scale = np.sqrt(2.0 / (in_channels * k_h * k_w))
        self.weight = np.random.randn(
            out_channels,
            in_channels,
            k_h,
            k_w
        ) * scale

        # Bias initialized to zeros
        self.bias = np.zeros(out_channels)

        # Store input for backward pass
        self.cache_input: Optional[np.ndarray] = None

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the convolution layer.

        :param x: Input tensor
        :type x: np.ndarray
        :return: Output feature maps
        :rtype: np.ndarray
        """

        self.cache_input = x # Store input for backward

        N, C_in, H, W = x.shape
        k_h, k_w = self.kernel_size

        # Compute output spatial dimensions
        H_out = (H + 2 * self.padding - k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - k_w) // self.stride + 1

        # Initialize output tensor
        output = np.zeros((N, self.out_channels, H_out, W_out))

        return output

    def backward(self, dL_dout: np.ndarray, lr: float) -> np.ndarray:
        """
        Compute the backward pass of the convolution layer.

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
        dL_dfilters = np.zeros_like(self.weight) # gradient w.r.t weights
        dL_dbias = np.zeros_like(self.bias) # gradient w.r.t bias

        # Loop over output channels
        for i in range(self.out_channels):

            # Compute gradient of loss w.r.t current filter
            dL_dfilters[i] = correlate2d(
                x[0, 0],
                dL_dout[0, i],
                mode="valid"
            )

            # Compute gradient of loss w.r.t input
            dL_dinput[0, 0] += correlate2d(
                dL_dout[0, i],
                self.weight[i, 0],
                mode="full"
            )

            # Compute gradient of loss w.r.t bias
            dL_dbias[i] = np.sum(dL_dout[:, i])
        
        # Store gradients for optimizer
        self.dL_dweight = dL_dfilters
        self.dL_dbias = dL_dbias

        # Update layer parameters
        self.weight -= lr * dL_dfilters
        self.bias -= lr * dL_dbias

        return dL_dinput