import numpy as np

from src.layers.batch_norm_layer import BatchNormLayer


def test_forward():
    """
    Test the forward pass of the BatchNormLayer.

    Verify that the forward pass produces an output tensor
    with the same shape as the input. Test for both single-image
    and batch inputs.
    """

    channels = 3
    height = 4
    width = 5

    # Initialize batch normalization layer
    bn = BatchNormLayer(num_channels=channels)

    # ----- Single image -----
    x_single = np.random.randn(channels, height, width)

    # Perform and vertify forward pass
    output_single = bn.forward(x_single)
    assert output_single.shape == x_single.shape

    # ----- Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, channels, height, width)

    output_batch = bn.forward(x_batch)
    assert output_batch.shape == x_batch.shape


def test_backward():
    """
    Test the backward pass of the BatchNormLayer.

    Verify that the backward pass returns a gradient with the same
    shape as the input and that gamma and beta parameters are updated.
    Test for both single-image
    and batch inputs.
    """

    channels = 3
    height = 4
    width = 5

    bn = BatchNormLayer(num_channels=channels)

    # ----- Single image -----
    x_single = np.random.randn(channels, height, width)

    output_single = bn.forward(x_single)

    # Random gradient from next layer
    dL_dout_single = np.random.randn(*output_single.shape)
    dL_dinput_single = bn.backward(dL_dout_single, lr=0.01)

    assert dL_dinput_single.shape == x_single.shape

    # ----- Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, channels, height, width)

    output_batch = bn.forward(x_batch)

    # Store learnable parameters before next update instance
    gamma_before = bn.gamma.copy()
    beta_before = bn.beta.copy()

    dL_dout_batch = np.random.randn(*output_batch.shape)
    dL_dinput_batch = bn.backward(dL_dout_batch, lr=0.01)

    assert dL_dinput_batch.shape == x_batch.shape
    assert not np.allclose(gamma_before, bn.gamma)
    assert not np.allclose(beta_before, bn.beta)
