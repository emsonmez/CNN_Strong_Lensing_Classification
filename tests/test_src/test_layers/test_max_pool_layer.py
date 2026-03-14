import numpy as np
from src.layers.max_pool_layer import MaxPoolLayer


def test_forward_maxpool():
    """
    Test the forward pass of MaxPoolLayer.
    """

    # Create a simple input tensor (C=1, H=4, W=4)
    x = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]]])

    pool = MaxPoolLayer(pool_size=2, stride=2)
    out = pool.forward(x)

    # Expected output after 2x2 max pooling
    expected = np.array([[[6, 8],
                          [14, 16]]])

    assert np.array_equal(out, expected), f"Forward pass output incorrect: {out}"


def test_backward_maxpool():
    """
    Test the backward pass of MaxPoolLayer.
    """

    # Input tensor (C=1, H=4, W=4)
    x = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]]])

    pool = MaxPoolLayer(pool_size=2, stride=2)
    pool.forward(x)

    # Gradient coming from next layer (same shape as output)
    dL_dout = np.array([[[1, 2],
                         [3, 4]]])
    dL_dinput = pool.backward(dL_dout)

    # Expected gradient propagated to max positions
    expected_grad = np.array([[[0, 0, 0, 0],
                               [0, 1, 0, 2],
                               [0, 0, 0, 0],
                               [0, 3, 0, 4]]])

    assert np.array_equal(dL_dinput, expected_grad), f"Backward pass gradient incorrect: {dL_dinput}"