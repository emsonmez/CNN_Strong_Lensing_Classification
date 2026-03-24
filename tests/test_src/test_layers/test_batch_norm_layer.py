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

    # ----- Conv: Single image -----
    x_single = np.random.randn(channels, height, width)

    output_single_train = bn.forward(x_single, training=True)
    output_single_eval = bn.forward(x_single, training=False)

    assert output_single_train.shape == x_single.shape
    assert output_single_eval.shape == x_single.shape

    # ----- Conv: Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, channels, height, width)

    output_batch_train = bn.forward(x_batch, training=True)
    output_batch_eval = bn.forward(x_batch, training=False)

    assert output_batch_train.shape == x_batch.shape
    assert output_batch_eval.shape == x_batch.shape

    # Outputs should differ (different stats)
    assert not np.allclose(output_batch_train, output_batch_eval)

    # ----- Dense: Single sample -----
    x_single_dense = np.random.randn(channels)

    output_single_dense_train = bn.forward(x_single_dense, training=True)
    output_single_dense_eval = bn.forward(x_single_dense, training=False)

    assert output_single_dense_train.shape == x_single_dense.shape
    assert output_single_dense_eval.shape == x_single_dense.shape

    # ----- Dense: Batch input -----
    x_batch_dense = np.random.randn(batch_size, channels)

    output_batch_dense_train = bn.forward(x_batch_dense, training=True)
    output_batch_dense_eval = bn.forward(x_batch_dense, training=False)

    assert output_batch_dense_train.shape == x_batch_dense.shape
    assert output_batch_dense_eval.shape == x_batch_dense.shape

    assert not np.allclose(output_batch_dense_train, output_batch_dense_eval)


def test_backward():
    """
    Test the backward pass of the BatchNormLayer.

    Verify that the backward pass returns a gradient with the same
    shape as the input and that gamma and beta parameters are computed.
    Test for both single-image and batch inputs.
    """

    channels = 3
    height = 4
    width = 5

    bn = BatchNormLayer(num_channels=channels)

    # ----- Conv: Single image -----
    x_single = np.random.randn(channels, height, width)

    output_single = bn.forward(x_single)

    # Random gradient from next layer
    dL_dout_single = np.random.randn(*output_single.shape)
    dL_dinput_single = bn.backward(dL_dout_single, lr=0.01)

    assert dL_dinput_single.shape == x_single.shape
    assert np.all(np.isfinite(dL_dinput_single))

    # ----- Conv: Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, channels, height, width)

    output_batch = bn.forward(x_batch)

    dL_dout_batch = np.random.randn(*output_batch.shape)
    dL_dinput_batch = bn.backward(dL_dout_batch, lr=0.01)

    assert dL_dinput_batch.shape == x_batch.shape
    assert np.all(np.isfinite(dL_dinput_batch))

    # Check gradients exist
    assert hasattr(bn, "dL_dgamma")
    assert hasattr(bn, "dL_dbeta")
    assert bn.dL_dgamma.shape == (channels,)
    assert bn.dL_dbeta.shape == (channels,)

    # ----- Dense: Single sample -----
    x_single_dense = np.random.randn(channels)

    output_single_dense = bn.forward(x_single_dense)

    dL_dout_single_dense = np.random.randn(*output_single_dense.shape)
    dL_dinput_single_dense = bn.backward(dL_dout_single_dense, lr=0.01)

    assert dL_dinput_single_dense.shape == x_single_dense.shape
    assert np.all(np.isfinite(dL_dinput_single_dense))

    # ----- Dense: Batch input -----
    x_batch_dense = np.random.randn(batch_size, channels)

    output_batch_dense = bn.forward(x_batch_dense)

    dL_dout_batch_dense = np.random.randn(*output_batch_dense.shape)
    dL_dinput_batch_dense = bn.backward(dL_dout_batch_dense, lr=0.01)

    assert dL_dinput_batch_dense.shape == x_batch_dense.shape
    assert np.all(np.isfinite(dL_dinput_batch_dense))

    # Check gradients exist
    assert bn.dL_dgamma.shape == (channels,)
    assert bn.dL_dbeta.shape == (channels,)
