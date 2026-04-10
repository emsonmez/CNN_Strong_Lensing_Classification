import numpy as np
from src.model.loss import CrossEntropyLoss


def test_forward():
    """
    Test the forward pass of the CrossEntropyLoss.

    Verify that the forward pass computes a valid scalar loss
    and remains numerically stable for both one-hot encoded (2D)
    and integer class index (1D) label inputs.
    """

    loss_fn = CrossEntropyLoss()

    # Simulated predictions (after softmax; rows randomly add up to 1)
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    # Case 1: 2D one-hot encoded labels; standard path
    y_true_onehot = np.array([[1, 0, 0], [0, 0, 1]])  # One-hot encoded true labels
    loss_onehot = loss_fn.forward(y_pred, y_true_onehot)

    # Loss should be a scalar and positive
    assert isinstance(loss_onehot, float)
    assert loss_onehot > 0

    # Case 2: 1D integer class indices
    y_true_labels = np.array([0, 2])
    loss_labels = loss_fn.forward(y_pred, y_true_labels)
    assert isinstance(loss_labels, float)
    assert loss_labels > 0

    # Both inputs represent the same labels so losses should be equal
    assert np.isclose(loss_onehot, loss_labels)


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
