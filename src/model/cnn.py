from typing import List
import numpy as np

# Imnporting all layers
from src.layers.activation_layer import ActivationLayer
from src.layers.batch_norm_layer import BatchNormLayer
from src.layers.conv_layer import ConvLayer
from src.layers.dense_layer import DenseLayer
from src.layers.dropout_layer import DropoutLayer
from src.layers.flatten_layer import FlattenLayer
from src.layers.max_pool_layer import MaxPoolLayer


class CNNModel:
    """
    Convolutional Neural Network (CNN) model and flowchart.
    """

    def __init__(self):
        """
        Initialize CNN model architecture.

        TODO: Optimize the hyper-parameters and order of layers if needed.
        """

        self.layers: List = [
            # First Conv Block
            ConvLayer(in_channels=1, out_channels=8, kernel_size=(3, 3)),
            BatchNormLayer(num_channels=8),
            ActivationLayer(alpha=0.01),
            MaxPoolLayer(pool_size=2, stride=2),
            # Second Conv Block
            ConvLayer(in_channels=8, out_channels=16, kernel_size=(3, 3)),
            BatchNormLayer(num_channels=16),
            ActivationLayer(alpha=0.01),
            MaxPoolLayer(pool_size=2, stride=2),
            # Third Conv Block (no pooling to avoid overfitting)
            ConvLayer(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            BatchNormLayer(num_channels=32),
            ActivationLayer(alpha=0.01),
            # Transition to Dense
            FlattenLayer(),
            # Dense Block
            DenseLayer(input_size=32 * 26 * 26, output_size=128),
            BatchNormLayer(num_channels=128),
            ActivationLayer(alpha=0.01),
            DropoutLayer(dropout_rate=0.15),
            # Output Layer
            # 7 classes (lensed quasar, supernoave, galaxy, their non-lensed versions,
            # and a "other/unidentifiable" case; might decide to drop this last case)
            DenseLayer(input_size=128, output_size=7),
        ]

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute the forward passes through all layers.

        :param x: Input data
        :type x: np.ndarray
        :param training: Whether model is in training mode
        :type training: bool
        :return: Output predictions
        :rtype: np.ndarray
        """

        for layer in self.layers:
            # Handle dropout separately (needs training flag)
            if isinstance(layer, DropoutLayer):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)

        return x

    def backward(self, dL: np.ndarray, lr: float) -> None:
        """
        Compute the backward passes through all layers.

        :param dL: Gradient of loss
        :type dL: np.ndarray
        :param lr: Learning rate (set to 0 if using optimizer)
        :type lr: float
        """

        for layer in reversed(self.layers):
            # Some layers don't take lr (like pooling, activation)
            try:
                dL = layer.backward(dL, lr)
            except TypeError:
                dL = layer.backward(dL)
