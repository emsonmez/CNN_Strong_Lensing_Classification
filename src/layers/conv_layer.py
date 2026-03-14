import numpy as np
from scipy.signal import correlate2d
from typing import Optional, Tuple


import numpy as np
from typing import Optional, Tuple
from scipy.signal import correlate2d


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

        scale = np.sqrt(2.0 / (in_channels * k_h * k_w))

        self.weight = np.random.randn(
            out_channels,
            in_channels,
            k_h,
            k_w
        ) * scale

        self.bias = np.zeros(out_channels)

        self.cache_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass of the convolution layer.

        :param x: Input tensor
        :type x: np.ndarray
        :return: Output feature maps
        :rtype: np.ndarray
        """

        self.cache_input = x

        N, C_in, H, W = x.shape
        k_h, k_w = self.kernel_size

        H_out = (H + 2 * self.padding - k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - k_w) // self.stride + 1

        output = np.zeros((N, self.out_channels, H_out, W_out))

        return output

    def backward(self, dL_dout: np.ndarray, lr: float) -> np.ndarray:
        """
        Perform the backward pass of the convolution layer.

        :param dL_dout: Gradient of the loss with respect to the layer output
        :type dL_dout: np.ndarray
        :param lr: Learning rate for parameter updates
        :type lr: float
        :return: Gradient of the loss with respect to the input
        :rtype: np.ndarray
        """

        x = self.cache_input

        dL_dinput = np.zeros_like(x)
        dL_dfilters = np.zeros_like(self.weight)
        dL_dbias = np.zeros_like(self.bias)

        for i in range(self.out_channels):

            # gradient w.r.t. filters
            dL_dfilters[i] = correlate2d(
                x[0, 0],
                dL_dout[0, i],
                mode="valid"
            )

            # gradient w.r.t. input
            dL_dinput[0, 0] += correlate2d(
                dL_dout[0, i],
                self.weight[i, 0],
                mode="full"
            )

            # gradient w.r.t. bias
            dL_dbias[i] = np.sum(dL_dout[:, i])

        # update parameters
        self.weight -= lr * dL_dfilters
        self.bias -= lr * dL_dbias

        return dL_dinput