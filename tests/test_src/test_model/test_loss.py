import numpy as np

from src.model.loss import CrossEntropyLoss


def test_forward():
    """
    Test the forward pass of the CrossEntropyLoss.

    Verify that the forward pass computes a valid scalar loss
    and remains numerically stable.
    """

    # Simulated predictions (after softmax; rows randomly add up to 1)
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    # One-hot encoded true labels
    y_true = np.array([[1, 0, 0], [0, 0, 1]])

    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(y_pred, y_true)

    # Loss should be a scalar and positive
    assert isinstance(loss, float)
    assert loss > 0


def test_backward():
    """
    Test the backward pass of the CrossEntropyLoss.

    Verify that gradients have the correct shape and follow
    the expected behavior.
    """

    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    y_true = np.array([[1, 0, 0], [0, 0, 1]])

    loss_fn = CrossEntropyLoss()
    loss_fn.forward(y_pred, y_true)

    # Number of samples derived dynamically
    N = y_pred.shape[0]

    dL_dz = loss_fn.backward()

    # Check shape consistency
    assert dL_dz.shape == y_pred.shape

    # Expected gradient
    expected = (y_pred - y_true) / N

    assert np.allclose(dL_dz, expected)
