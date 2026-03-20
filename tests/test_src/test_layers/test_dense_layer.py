import numpy as np
from src.layers.dense_layer import DenseLayer


def test_forward():
    """
    Test the forward pass of the DenseLayer.

    Verify that the forward pass produces an output vector of the expected size
    and that the softmax probabilities sum to one. Test for both single-image 
    and batch inputs.
    """

    input_size = 8
    output_size = 3

    # Initialize dense layer
    dense = DenseLayer(
        input_size=input_size,
        output_size=output_size
    )

    # ----- Single image -----
    x_single = np.random.randn(input_size)

    output_single = dense.forward(x_single)

    # Verify output shape
    assert output_single.shape == (1, output_size)

    # Verify softmax probabilities sum to 1
    sums = np.sum(output_single, axis=1)
    assert np.allclose(sums, 1.0)

    # ----- Batch input -----
    batch_size = 2
    x_batch = np.random.randn(batch_size, input_size)

    output_batch = dense.forward(x_batch)

    assert output_batch.shape == (batch_size, output_size)

    sums_batch = np.sum(output_batch, axis=1)
    assert np.allclose(sums_batch, 1.0)


def test_backward():
    """
    Test the backward pass of the DenseLayer.

    Verify that the backward pass returns a gradient vector with the same
    shape as the input and that the layer parameters are updated. Test 
    for both single-image and batch inputs.
    """

    input_size = 8
    output_size = 3

    dense = DenseLayer(
        input_size=input_size,
        output_size=output_size
    )
    
    # ----- Single image -----
    x_single = np.random.randn(input_size)

    output_single = dense.forward(x_single)

    # Random gradient from next layer
    dL_dout_single = np.random.randn(output_size)
    dL_dinput_single = dense.backward(dL_dout_single, lr=0.01)

    assert dL_dinput_single.shape == (1, input_size)

    
    # ----- Batch input -----
    batch_size  = 2
    x_batch = np.random.randn(batch_size, input_size)
    output_batch = dense.forward(x_batch)

    dL_dout_batch = np.random.randn(batch_size, output_size)
    
    # Store weights before backward pass
    weights_before = dense.weight.copy()
    bias_before = dense.bias.copy()

    dL_dinput_batch = dense.backward(dL_dout_batch, lr=0.01)
    
    # Verify gradient shape
    assert dL_dinput_batch.shape == x_batch.shape

    # Verify parameters were updates
    assert not np.allclose(weights_before, dense.weight)
    assert not np.allclose(bias_before, dense.bias)

    # Verify gradients were stored (for optimizer)
    assert hasattr(dense, "dL_dweight")
    assert hasattr(dense, "dL_dbias")