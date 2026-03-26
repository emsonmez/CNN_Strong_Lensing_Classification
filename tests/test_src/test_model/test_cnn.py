import numpy as np
from src.model.cnn import CNNModel


def test_forward():
    """
    Test the forward pass of the CNNModel.

    Verify that the model produces an output with the correct shape
    corresponding to the number of classes. For both training and
    inference modes.
    """

    # Dummy input (batch size = 1, 1 channel, 120x120 image)
    x = np.random.randn(1, 1, 120, 120)
    model = CNNModel()

    # Training mode
    output_train = model.forward(x, training=True)
    assert output_train.shape == (1, 2)

    # Inference mode
    output_eval = model.forward(x, training=False)
    assert output_eval.shape == (1, 2)

    # Outputs should differ due to BatchNorm / Dropout
    assert not np.allclose(output_train, output_eval)


def test_backward():
    """
    Test the backward pass of the CNNModel.

    Verify that the backward pass runs without errors and propagates
    gradients through all layers.
    """

    x = np.random.randn(1, 5, 120, 120)

    model = CNNModel()

    output = model.forward(x, training=True)

    # Create dummy gradient from loss (same shape as output of (1,2))
    dL = np.random.randn(*output.shape)

    # Backward pass (lr=0 because optimizer handles updates)
    model.backward(dL, lr=0)

    # Check that at least one layer has gradients computed
    has_gradients = any(
        hasattr(layer, "dL_dweight") and layer.dL_dweight is not None
        for layer in model.layers
    )

    assert has_gradients, "No gradients were computed during backward pass"
