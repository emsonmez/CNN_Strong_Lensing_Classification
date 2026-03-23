import numpy as np
from src.layers.conv_layer import ConvLayer


def test_forward():
    """
    Test the forward pass of the ConvolutionLayer.

    Verify correct output shape. Test for both batch and single-image inputs,
    and with padding/non-padding to cover all code branches.
    """

    in_channels = 3
    height = 10
    width = 10

    # ----- Without padding -----
    conv = ConvLayer(
        in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=0
    )
    expected_h = height - 3 + 1
    expected_w = width - 3 + 1

    # ----- Single image -----
    x_single = np.random.randn(in_channels, height, width)
    output_single = conv.forward(x_single)
    assert output_single.shape == (4, expected_h, expected_w)

    # ----- Batch input -----
    batch = 2
    x_batch = np.random.randn(batch, in_channels, height, width)
    output_batch = conv.forward(x_batch)
    assert output_batch.shape == (batch, 4, expected_h, expected_w)

    # ----- With padding -----
    conv_pad = ConvLayer(
        in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=2
    )
    # Output dimensions increase due to padding
    expected_h_pad = height + 2 * 2 - 3 + 1
    expected_w_pad = width + 2 * 2 - 3 + 1

    x_single_pad = np.random.randn(in_channels, height, width)
    output_single_pad = conv_pad.forward(x_single_pad)
    assert output_single_pad.shape == (4, expected_h_pad, expected_w_pad)

    x_batch_pad = np.random.randn(batch, in_channels, height, width)
    output_batch_pad = conv_pad.forward(x_batch_pad)
    assert output_batch_pad.shape == (batch, 4, expected_h_pad, expected_w_pad)


def test_backward():
    """
    Test the backward pass of the ConvolutionLayer.

    Verify gradient shape and parameter updates. Test for
    both batch and single inputs, and with padding/non-padding to cover
    all code branches.
    """

    in_channels = 3
    height = 10
    width = 10

    # ----- Without padding -----
    conv = ConvLayer(
        in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=0
    )

    # ----- Single image -----
    x_single = np.random.randn(in_channels, height, width)
    output_single = conv.forward(x_single)
    dL_dout_single = np.random.randn(*output_single.shape)
    dL_dinput_single = conv.backward(dL_dout_single, lr=0.01)

    assert dL_dinput_single.shape == x_single.shape

    # ----- Batch input -----
    batch = 2
    x_batch = np.random.randn(batch, in_channels, height, width)
    output_batch = conv.forward(x_batch)
    dL_dout_batch = np.random.randn(*output_batch.shape)
    weights_before = conv.weight.copy()
    dL_dinput_batch = conv.backward(dL_dout_batch, lr=0.01)

    assert dL_dinput_batch.shape == x_batch.shape
    assert not np.allclose(weights_before, conv.weight)

    # ----- With padding -----
    conv_pad = ConvLayer(
        in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=2
    )
    x_batch_pad = np.random.randn(batch, in_channels, height, width)
    output_batch_pad = conv_pad.forward(x_batch_pad)
    dL_dout_batch_pad = np.random.randn(*output_batch_pad.shape)
    weights_before_pad = conv_pad.weight.copy()
    dL_dinput_batch_pad = conv_pad.backward(dL_dout_batch_pad, lr=0.01)

    assert dL_dinput_batch_pad.shape == x_batch_pad.shape
    assert not np.allclose(weights_before_pad, conv_pad.weight)
