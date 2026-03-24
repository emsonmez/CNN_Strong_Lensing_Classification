from typing import List
from inspect import signature
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

    def __init__(self, input_shape=(1, 120, 120)):
        """
        Initialize CNN model architecture.

        :param input_shape: Shape of input image (C, H, W)
        :type input_shape: tuple
        """

        # Convolutional backbone
        self.feature_extractor: List = [
            ConvLayer(in_channels=1, out_channels=8, kernel_size=(3, 3)),
            BatchNormLayer(num_channels=8),
            ActivationLayer(alpha=0.01),
            MaxPoolLayer(pool_size=2, stride=2),
            ConvLayer(in_channels=8, out_channels=16, kernel_size=(3, 3)),
            BatchNormLayer(num_channels=16),
            ActivationLayer(alpha=0.01),
            MaxPoolLayer(pool_size=2, stride=2),
            ConvLayer(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            BatchNormLayer(num_channels=32),
            ActivationLayer(alpha=0.01),
        ]

        # Compute flatten size dynamically
        dummy = np.random.randn(1, *input_shape)  # batch_size=(1, C, H, W)
        x = dummy

        for layer in self.feature_extractor:
            sig = signature(layer.forward)

            if "training" in sig.parameters:
                x = layer.forward(x, training=True)
            else:
                x = layer.forward(x)

        self.flatten_size = x.size // x.shape[0]

        # Rest of the full model
        self.layers: List = [
            *self.feature_extractor,
            FlattenLayer(),
            # Dense Block
            DenseLayer(input_size=self.flatten_size, output_size=128),
            BatchNormLayer(num_channels=128),
            ActivationLayer(alpha=0.01),
            DropoutLayer(dropout_rate=0.15),
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
            sig = signature(layer.forward)
            if "training" in sig.parameters:
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)

        return x

    def backward(self, dL: np.ndarray, lr: float = 0.0) -> None:
        """
        Compute backward passes through all layers and update trainable parameters.

        Pass `lr` only to layers that accept it.

        :param dL: Gradient of loss
        :type dL: np.ndarray
        :param lr: Learning rate (unused when using optimizer)
        :type lr: float
        """

        for layer in reversed(self.layers):
            # Check if backward method has a 'lr' parameter
            from inspect import signature

            sig = signature(layer.backward)
            if "lr" in sig.parameters:
                dL = layer.backward(dL, lr)
            else:
                dL = layer.backward(dL)
