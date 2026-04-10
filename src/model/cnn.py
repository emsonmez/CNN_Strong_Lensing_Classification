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

    def __init__(
        self,
        input_shape=(5, 120, 120),
        conv_channels=(8, 16, 32),
        kernel_size=3,
        pool_size=2,
        alpha=0.01,
        dropout_rate=0.15,
        hidden_size=128,
    ):
        """
        Initialize CNN model architecture.

        :param input_shape: Shape of input tensor image (C, H, W)
        :type input_shape: tuple
        :param conv_channels: Convolution layer channels
        :type conv_channels: tuple
        :param kernel_size: Convolution kernel size
        :type kernel_size: int
        :param pool_size: Max pooling size (Note that pool_size = stride)
        :type pool_size: int
        :param alpha: LeakyReLU Activation slope
        :type alpha: float
        :param dropout_rate: Dropout rate
        :type dropout_rate: float
        :param hidden_size: Dense layer size
        :type hidden_size: int
        """
        # Extract input channels dynamically
        in_channels = input_shape[0]
        c1, c2, c3 = conv_channels  # Conv batch layers levels 1,2,3

        # Convolutional backbone
        self.feature_extractor: List = [
            # 1: Single Channel, 5: Multi-Class for each Pan-STARRS1 filter (g,r,i,z,y)
            ConvLayer(in_channels, c1, (kernel_size, kernel_size)),
            BatchNormLayer(c1),
            ActivationLayer(alpha),
            MaxPoolLayer(pool_size, pool_size),
            ConvLayer(c1, c2, (kernel_size, kernel_size)),
            BatchNormLayer(c2),
            ActivationLayer(alpha),
            MaxPoolLayer(pool_size, pool_size),
            ConvLayer(c2, c3, (kernel_size, kernel_size)),
            BatchNormLayer(c3),
            ActivationLayer(alpha),
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

        self.flatten_size = x.reshape(1, -1).shape[1]

        # Rest of the full model
        self.layers: List = [
            *self.feature_extractor,
            FlattenLayer(),
            # Dense Block
            DenseLayer(self.flatten_size, hidden_size),
            BatchNormLayer(hidden_size),
            ActivationLayer(alpha),
            DropoutLayer(dropout_rate),
            # Binary classification (Lensed or Nonlensed)
            DenseLayer(hidden_size, 2),
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
