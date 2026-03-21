import numpy as np
from src.layers.dropout_layer import DropoutLayer


def test_forward():
    """
    Test the forward pass of the DropoutLayer.

    Vertify that the forward pass preserves input shape
    ,zeroes out some values during training, and leaves
    input unchanged during inference.
    """

    # Input tensor
    x = np.ones((3, 4, 5))

    dropout = DropoutLayer(dropout_rate=0.3)

    # Forward pass (training mode)
    output_train = dropout.forward(x, training=True)

    assert output_train.shape == x.shape
    assert np.any(output_train == 0)

    # Forward pass (inference mode)
    output_eval = dropout.forward(x, training=False)

    assert np.allclose(output_eval, x)


def test_backward():
    """
    Test the backward pass of the DropoutLayer.

    Verify that gradients have the same shape as input,
    are zero where activations were dropped, and remain
    unchanged when no dropout mask is applied.
    """

    x = np.ones((3, 4, 5))

    # Gradient from next layer
    dL_dout = np.ones_like(x)

    # No forward pass (mask is None)
    dropout_no_train = DropoutLayer(dropout_rate=0.3)
    dL_identity = dropout_no_train.backward(dL_dout)

    assert np.allclose(dL_identity, dL_dout)

    # Dropout mask (training mode)
    dropout = DropoutLayer(dropout_rate=0.5)

    dropout.forward(x, training=True)

    # Backward pass
    dL_dinput = dropout.backward(dL_dout)

    assert dL_dinput.shape == x.shape
    assert np.any(dL_dinput == 0)
