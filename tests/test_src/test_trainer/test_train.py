import numpy as np
from src.model.cnn import CNNModel
from src.model.loss import CrossEntropyLoss
from src.model.optimizer import AdamOptimizer
from src.trainer.train import Trainer


def test_train():
    """
    Test the Trainer class.

    Vertify training runs without errors,
    history is correctly returnes returns epoch-level loss,
    accuracy, and batch loss, loss values are valid (positive),
    and model weights are updated. Generalized for one-hot (2D) and integer
    (1D) label inputs for both y and y_val, and the no-validation
    else branch.
    """

    # Fake training and validation data
    X_train = np.random.randn(5, 5, 28, 28)  # training samples
    y_train = np.eye(2)[np.random.randint(0, 2, size=5)]  # one-hot labels

    X_val = np.random.randn(4, 5, 28, 28)  # validation samples
    y_val = np.eye(2)[np.random.randint(0, 2, size=4)]

    model = CNNModel(input_shape=(5, 28, 28))
    loss_fn = CrossEntropyLoss()
    optimizer = AdamOptimizer(lr=0.001)

    # Store initial weights
    initial_weights = []
    for layer in model.layers:
        if hasattr(layer, "weight"):
            initial_weights.append(layer.weight.copy())

    trainer = Trainer(model, loss_fn, optimizer)

    # Case 1: 2D one-hot encoded y and y_val; standard path
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=5)

    # Check history dictionary keys
    assert "epoch_loss" in history
    assert "epoch_accuracy" in history
    assert "batch_loss" in history
    assert "val_loss" in history  # empty if no validation
    assert "val_accuracy" in history  # empty if no validation

    # Check length of history lists
    assert len(history["epoch_loss"]) == 2
    assert len(history["epoch_accuracy"]) == 2
    assert len(history["batch_loss"]) >= 1  # At least one batch per epoch
    assert len(history["val_loss"]) == 2
    assert len(history["val_accuracy"]) == 2

    # Check loss values are positive
    assert all(loss > 0 for loss in history["epoch_loss"])
    assert all(batch_loss > 0 for batch_loss in history["batch_loss"])
    assert all(val_loss > 0 for val_loss in history["val_loss"])

    # Case 2: 1D integer class indices for y_value
    y_train_labels = np.random.randint(0, 2, size=5)
    y_val_labels = np.random.randint(0, 2, size=4)

    history_labels = trainer.train(
        X_train, y_train_labels, X_val, y_val_labels, epochs=1, batch_size=5
    )

    assert len(history_labels["epoch_loss"]) == 1
    assert len(history_labels["val_loss"]) == 1

    assert all(loss > 0 for loss in history_labels["epoch_loss"])
    assert all(val_loss > 0 for val_loss in history_labels["val_loss"])

    # Case 3: No validation data; for the else branch
    history_no_val = trainer.train(X_train, y_train, epochs=1, batch_size=5)

    # Check that val_loss and val_accuracy remain empty
    assert history_no_val["val_loss"] == []
    assert history_no_val["val_accuracy"] == []

    # Also check if these are still populated
    assert len(history_no_val["epoch_loss"]) == 1
    assert len(history_no_val["epoch_accuracy"]) == 1
    assert len(history_no_val["batch_loss"]) >= 1  # at least one batch

    # Check weights updated across all training calls
    updated_weights = []
    for layer in model.layers:
        if hasattr(layer, "weight"):
            updated_weights.append(layer.weight)

    assert any(
        not np.allclose(w0, w1) for w0, w1 in zip(initial_weights, updated_weights)
    )
