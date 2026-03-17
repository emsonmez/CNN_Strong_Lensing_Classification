import numpy as np 
from src.layers.batch_norm_layer import BatchNormLayer


def test_forward():
    """
    Test the forward pass of the BatchNormLayer.

    Verify that the forward pass produces an output tensor
    with the same shape as the input.
    """

    channels = 3
    height = 4
    width = 5

    # Create random input tensor
    x = np.random.randn(channels, height, width)

    # Initialize batch normalization layer
    bn = BatchNormLayer(num_channels=channels)

    # Perform and vertify forward pass
    output = bn.forward(x)
    assert output.shape == x.shape

def test_backward():
    """
    Test the backward pass of the BatchNormLayer.

    Verify that the backward pass returns a gradient with the same
    shape as the input and that gamma and beta parameters are updated.
    """

    channels = 3
    height = 4
    width = 5

    x = np.random.randn(channels, height, width)

    bn = BatchNormLayer(num_channels=channels)

    output = bn.forward(x)

    # Random gradient from next layer
    dL_dout = np.random.randn(*output.shape)

    # Store learnable parameters before next update instance
    gamma_before = bn.gamma.copy()
    beta_before = bn.beta.copy()

    dL_dinput = bn.backward(dL_dout, lr=0.01)

    assert dL_dinput.shape == x.shape
    assert not np.allclose(gamma_before, bn.gamma)
    assert not np.allclose(beta_before, bn.beta)
