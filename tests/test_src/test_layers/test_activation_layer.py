import numpy as np
from src.layers.activation_layer import ActivationLayer


def test_forward():
    """
    Test the forward pass of the LeakyReLULayer.

    Verify that the forward pass correctly applies the Leaky ReLU
    activation function to the input tensor.
    """

    # Create input tensor with both positive and negative values
    x = np.array([[[-1.0, 2.0], [3.0, -4.0]]])

    # Initialize activation layer
    relu = ActivationLayer(alpha=0.01)

    output = relu.forward(x)

    # Expected output after Leaky ReLU
    expected = np.array([[[-0.01, 2.0], [3.0, -0.04]]])

    assert np.allclose(output, expected)


def test_backward():
    """
    Test the backward pass of the LeakyReLULayer.

    Verify that the backward pass correctly propagates gradients
    based on the Leaky ReLU derivative.
    """

    # Input tensor
    x = np.array([[[-1.0, 2.0], [3.0, -4.0]]])

    relu = ActivationLayer(alpha=0.01)

    # Run forward pass to cache input
    relu.forward(x)

    # Gradient from next layer
    dL_dout = np.ones_like(x)

    dL_dinput = relu.backward(dL_dout)

    # Expected gradient (alpha for negative, 1 for positive)
    expected_grad = np.array([[[0.01, 1.0], [1.0, 0.01]]])

    assert np.allclose(dL_dinput, expected_grad)
