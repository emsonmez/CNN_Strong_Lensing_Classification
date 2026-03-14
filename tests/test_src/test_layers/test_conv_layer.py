import numpy as np
from src.layers.conv_layer import ConvLayer


def test_forward_output_shape():
    """
    Verify that the forward pass produces the expected output shape.
    """

    batch = 2
    in_channels = 3
    height = 10
    width = 10

    # Create random input tensor (batch, channels, height, width)
    x = np.random.randn(batch, in_channels, height, width)

    # Initialize convolutional layer
    conv = ConvLayer(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3),
        stride=1,
        padding=0
    )

    output = conv.forward(x)

    # Compute expected output spatial dimensions
    expected_h = height - 3 + 1
    expected_w = width - 3 + 1

    assert output.shape == (batch, 4, expected_h, expected_w)


def test_backward_shapes():
    """
    Verify that backward pass returns gradient with same shape as input.
    """

    batch = 2
    in_channels = 3
    height = 10
    width = 10

    x = np.random.randn(batch, in_channels, height, width)

    conv = ConvLayer(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3),
        stride=1,
        padding=0
    )

    output = conv.forward(x)

    # Random gradient coming from next layer
    dL_dout = np.random.randn(*output.shape)

    # store weights to check update
    weights_before = conv.weight.copy()

    dL_dinput = conv.backward(dL_dout, lr=0.01)

    assert dL_dinput.shape == x.shape
    assert not np.allclose(weights_before, conv.weight)