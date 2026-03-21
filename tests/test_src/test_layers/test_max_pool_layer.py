import numpy as np

from src.layers.max_pool_layer import MaxPoolLayer


def test_forward():
    """
    Test the forward pass of MaxPoolLayer.

    Verify that the forward pass returns the correct pooled values for a
    simple deterministic input tensor. Test for both single-image
    and batch inputs.
    """

    pool = MaxPoolLayer(pool_size=2, stride=2)

    # ----- Single image -----
    # Create a simple input tensor (C=1, H=4, W=4)
    x_single = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
    )

    output_single = pool.forward(x_single)

    # Expected output after 2x2 max pooling
    expected_single = np.array([[[6, 8], [14, 16]]])

    assert np.array_equal(
        output_single, expected_single
    ), f"Single forward incorrect: {output_single}"

    # ----- Batch input -----
    # Create a simple input tensor (N = 1, C=1, H=4, W=4)
    x_batch = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )

    output_batch = pool.forward(x_batch)

    expected_batch = np.array([[[[6, 8], [14, 16]]]])

    # Verify output matches expected pooled values
    assert np.array_equal(
        output_batch, expected_batch
    ), f"Batch forward incorrect: {output_batch}"


def test_backward():
    """
    Test the backward pass of MaxPoolLayer.

    Verify that the backward pass propagates gradients only to the
    positions that contained the maximum values during the forward pass.
    Test for both single-image and batch inputs.
    """

    pool = MaxPoolLayer(pool_size=2, stride=2)

    # ----- Single image -----
    # Input tensor (C=1, H=4, W=4)
    x_single = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
    )

    pool.forward(x_single)

    # Gradient coming from next layer (same shape as output)
    dL_dout_single = np.array([[[1, 2], [3, 4]]])
    dL_dinput_single = pool.backward(dL_dout_single)

    # Expected gradient propagated to max positions
    expected_single = np.array(
        [[[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]]
    )

    # Verify gradients were routed only to max locations
    assert np.array_equal(
        dL_dinput_single, expected_single
    ), f"Single backward incorrect: {dL_dinput_single}"

    # ----- Batch input -----
    # Input tensor (N = 1, C=1, H=4, W=4)
    x_batch = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )

    pool.forward(x_batch)

    dL_dout_batch = np.array([[[[1, 2], [3, 4]]]])
    dL_dinput_batch = pool.backward(dL_dout_batch)

    expected_batch = np.array(
        [[[[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]]]
    )

    assert np.array_equal(
        dL_dinput_batch, expected_batch
    ), f"Batch backward incorrect: {dL_dinput_batch}"
