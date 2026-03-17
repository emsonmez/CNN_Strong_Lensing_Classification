import numpy as np 
from src.layers.flatten_layer import FlattenLayer


def test_forward():
    """
    Test the forward pass of the FlattenLayer.

    Verify that the forward pass correctly flattens the input tensor
    into a one-dimensional feature vector.
    """

    channels = 3
    height = 4
    width = 5

    # Create random input tensor (C, H, W)
    x = np.random.randn(channels, height, width)

    flatten = FlattenLayer()

    output = flatten.forward(x)

    # Expected flattened size
    expected_size = channels * height * width

    assert output.shape == (expected_size,)

def test_backward():
    """
    Test the backward pass of the FlattenLayer.

    Verify that the backward pass reshapes the gradient back to the
    original input tensor dimensions.
    """

    channels = 3
    height = 4
    width = 5

    # Create random input tensor
    x = np.random.randn(channels, height, width)

    flatten = FlattenLayer()

    output = flatten.forward(x)

    # Random gradient coming from next layer
    dL_dout = np.random.randn(*output.shape)

    dL_dinput = flatten.backward(dL_dout)

    assert dL_dinput.shape == x.shape
