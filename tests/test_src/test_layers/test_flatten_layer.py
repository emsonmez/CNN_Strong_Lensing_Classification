import numpy as np
from src.layers.flatten_layer import FlattenLayer


def test_forward():
    """
    Test the forward pass of the FlattenLayer.

    Verify that the forward pass correctly flattens the input tensor
    into a one-dimensional feature vector. Test for both single-image
    and batch inputs.
    """

    channels = 3
    height = 4
    width = 5

    # ----- Single image -----
    x = np.random.randn(channels, height, width)

    flatten = FlattenLayer()

    output = flatten.forward(x)

    # Expected flattened size
    expected_size = channels * height * width

    assert output.shape == (expected_size,)

    # ----- Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, channels, height, width)

    output_batch = flatten.forward(x_batch)

    assert output_batch.shape == (batch_size, expected_size)


def test_backward():
    """
    Test the backward pass of the FlattenLayer.

    Verify that the backward pass reshapes the gradient back to the
    original input tensor dimensions. Test for both single-image
    and batch inputs.
    """

    channels = 3
    height = 4
    width = 5

    # ----- Single image -----
    x_single = np.random.randn(channels, height, width)

    flatten = FlattenLayer()

    output_single = flatten.forward(x_single)

    # Random gradient coming from next layer
    dL_dout = np.random.randn(*output_single.shape)

    dL_dinput = flatten.backward(dL_dout)

    assert dL_dinput.shape == x_single.shape

    # ----- Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, channels, height, width)

    flatten = FlattenLayer()
    output_batch = flatten.forward(x_batch)

    # Random gradient coming from next layer
    dL_dout_batch = np.random.randn(*output_batch.shape)
    dL_dinput_batch = flatten.backward(dL_dout_batch)

    assert dL_dinput_batch.shape == x_batch.shape
