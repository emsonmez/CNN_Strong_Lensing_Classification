import numpy as np
from src.layers.dense_layer import DenseLayer


def test_forward():
    """
    Test the forward pass of the DenseLayer.

    Verify that the forward pass produces an output vector of the expected size
    and that the softmax probabilities sum to one.
    """

    input_size = 8
    output_size = 3

    # Create random input feature vector
    x = np.random.randn(input_size)

    # Initialize dense layer
    dense = DenseLayer(
        input_size=input_size,
        output_size=output_size
    )

    output = dense.forward(x)

    # Verify output shape
    assert output.shape == (output_size,)

    # Verify softmax probabilities sum to 1
    assert np.isclose(np.sum(output), 1.0)


def test_backward():
    """
    Test the backward pass of the DenseLayer.

    Verify that the backward pass returns a gradient vector with the same
    shape as the input and that the layer parameters are updated.
    """

    input_size = 8
    output_size = 3
    x = np.random.randn(input_size)

    dense = DenseLayer(
        input_size=input_size,
        output_size=output_size
    )

    output = dense.forward(x)

    # Random gradient from next layer
    dL_dout = np.random.randn(output_size)

    # Store weights before backward pass
    weights_before = dense.weight.copy()

    dL_dinput = dense.backward(dL_dout, lr=0.01)
    
    assert dL_dinput.shape == x.shape

    # Verify parameters were updated
    assert not np.allclose(weights_before, dense.weight)