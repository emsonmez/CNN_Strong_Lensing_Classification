from typing import Optional, Tuple
import numpy as np


class ConvLayer:
    """
    A single 2D convolutional layer.

    Applies learnable filters across spatial regions of the input tensor.
    Produces one feature map per filter. Generalized for both single-image
    and batch inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
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
        self.weight = np.random.randn(out_channels, in_channels, k_h, k_w) * scale

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

        single_image = False

        if x.ndim == 3:  # (C, H, W)
            x = x[np.newaxis, :, :, :]
            single_image = True

        self.cache_input = x
        self.single_image = single_image

        (
            self.batch_size,
            self.num_channels,
            self.input_height,
            self.input_width,
        ) = x.shape

        # Initialize kernel dimensions
        kernel_height, kernel_width = self.kernel_size

        # Output spatial dimensions
        self.output_height = (
            self.input_height + 2 * self.padding - kernel_height
        ) // self.stride + 1
        self.output_width = (
            self.input_width + 2 * self.padding - kernel_width
        ) // self.stride + 1

        # Padding
        if self.padding > 0:
            # Pad input spatially (height & width)
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            x_padded = x  # No padding

        output = np.zeros(
            (
                self.batch_size,
                self.out_channels,
                self.output_height,
                self.output_width,
            )
        )

        # Loop over channels and output positions
        # TODO: Vectorize this
        for n in range(self.batch_size):
            for out_c in range(self.out_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.stride
                        end_i = start_i + kernel_height
                        start_j = j * self.stride
                        end_j = start_j + kernel_width

                        region = x_padded[n, :, start_i:end_i, start_j:end_j]

                        output[n, out_c, i, j] = (
                            np.sum(region * self.weight[out_c]) + self.bias[out_c]
                        )

        return output[0] if self.single_image else output

    def backward(self, dL_dout: np.ndarray, lr: float) -> np.ndarray:
        """
        Compute the backward pass of the convolution layer.

        :param dL_dout: Gradient of the loss with respect to the layer output
        :type dL_dout: np.ndarray
        :param lr: Learning rate
        :type lr: float
        :return: Gradient of the loss with respect to the input
        :rtype: np.ndarray
        """

        x = self.cache_input

        if self.single_image:
            dL_dout = dL_dout[np.newaxis, :, :, :]

        kernel_height, kernel_width = self.kernel_size

        # Padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
            )
            dL_dinput_padded = np.zeros_like(x_padded)
        else:
            x_padded = x
            dL_dinput_padded = np.zeros_like(x)

        # Initialize gradients
        dL_dweight = np.zeros_like(self.weight)
        dL_dbias = np.zeros_like(self.bias)

        # Loop over channels and output positions
        for n in range(self.batch_size):
            for out_c in range(self.out_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.stride
                        end_i = start_i + kernel_height
                        start_j = j * self.stride
                        end_j = start_j + kernel_width

                        region = x_padded[n, :, start_i:end_i, start_j:end_j]

                        # Compute gradient of loss w.r.t current filter
                        dL_dweight[out_c] += region * dL_dout[n, out_c, i, j]

                        # Compute gradient of loss w.r.t input
                        dL_dinput_padded[n, :, start_i:end_i, start_j:end_j] += (
                            self.weight[out_c] * dL_dout[n, out_c, i, j]
                        )

                        # Compute gradient of loss w.r.t bias
                        dL_dbias[out_c] += dL_dout[n, out_c, i, j]

        # Remove padding
        if self.padding > 0:
            dL_dinput = dL_dinput_padded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        else:
            dL_dinput = dL_dinput_padded

        # Store gradients for optimizer
        self.dL_dweight = dL_dweight
        self.dL_dbias = dL_dbias

        # Update layer parameters (only if lr > 0)
        self.weight -= lr * dL_dweight
        self.bias -= lr * dL_dbias

        return dL_dinput[0] if self.single_image else dL_dinput
