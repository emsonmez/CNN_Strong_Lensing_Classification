from typing import Optional
import numpy as np


class BatchNormLayer:
    """
    Batch normalization layer for stabilizing training.

    Normalizes feature maps using per-channel mean and variance,
    then applies learnable scale (gamma) and shift (beta).
    """

    def __init__(self, num_channels: int, epsilon: float = 1e-5):
        """
        Initialize batch normalization layer parameters.

        :param num_channels: Number of feature map channels
        :type num_channels: int
        :param epsilon: Small constant for numerical stability
        :type epsilon: float
        """

        self.num_channels = num_channels
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)

        # Stored for backward pass
        self.cache_input: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.variance: Optional[np.ndarray] = None
        self.x_hat: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the batch normalization layer.

        :param x: Input tensor of shape (C, H, W)
        :type x: np.ndarray
        :return: Normalized and scaled output tensor
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

        output = np.zeros_like(x)

        # Compute per-channel mean and variance
        self.mean = np.zeros(self.num_channels)
        self.variance = np.zeros(self.num_channels)
        self.x_hat = np.zeros_like(x)

        for c in range(self.num_channels):
            # Empirical mean and variance over spatial dimensions
            self.mean[c] = np.mean(x[:, c, :, :])
            self.variance[c] = np.var(x[:, c, :, :])

            # Normalize
            self.x_hat[:, c, :, :] = (x[:, c, :, :] - self.mean[c]) / np.sqrt(
                self.variance[c] + self.epsilon
            )

            # Scale and shift
            output[:, c, :, :] = self.gamma[c] * self.x_hat[:, c, :, :] + self.beta[c]

        return output[0] if self.single_image else output

    def backward(self, dL_dout: np.ndarray, lr: float) -> np.ndarray:
        """
        Compute the backward pass of the batch normalization layer.

        :param dL_dout: Gradient of the loss with respect to the layer output
        :type dL_dout: np.ndarray
        :param lr: Learning rate for parameter updates
        :type lr: float
        :return: Gradient of the loss with respect to the input
        :rtype: np.ndarray
        """

        x = self.cache_input

        if self.single_image:
            dL_dout = dL_dout[np.newaxis, :, :, :]

        self.batch_size, self.num_channels, self.input_height, self.input_width = (
            x.shape
        )

        # Total number of elements per channel
        N = self.batch_size * self.input_height * self.input_width

        # Initialize gradients
        dL_dinput = np.zeros_like(x)
        dL_dgamma = np.zeros_like(self.gamma)
        dL_dbeta = np.zeros_like(self.beta)

        for c in range(self.num_channels):
            # Step 1: gradients w.r.t gamma and beta
            dL_dgamma[c] = np.sum(dL_dout[:, c, :, :] * self.x_hat[:, c, :, :])
            dL_dbeta[c] = np.sum(dL_dout[:, c, :, :])

            # Step 2: gradient w.r.t normalized input
            dL_dxhat = dL_dout[:, c, :, :] * self.gamma[c]

            # Step 3: gradient w.r.t variance
            dL_dvar = np.sum(
                dL_dxhat
                * (x[:, c, :, :] - self.mean[c])
                * (-0.5)
                * (self.variance[c] + self.epsilon) ** (-1.5)
            )

            # Step 4: gradient w.r.t mean
            dL_dmean = np.sum(
                dL_dxhat * (-1 / np.sqrt(self.variance[c] + self.epsilon))
            ) + dL_dvar * (1 / N) * np.sum(-2 * (x[:, c, :, :] - self.mean[c]))

            # Step 5: gradient w.r.t input
            dL_dinput[:, c, :, :] = (
                dL_dxhat * (1 / np.sqrt(self.variance[c] + self.epsilon))
                + dL_dvar * (2 * (x[:, c, :, :] - self.mean[c]) / N)
                + dL_dmean * (1 / N)
            )

        # Update parameters
        self.gamma -= lr * dL_dgamma
        self.beta -= lr * dL_dbeta

        return dL_dinput[0] if self.single_image else dL_dinput
