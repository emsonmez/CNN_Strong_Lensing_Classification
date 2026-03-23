import numpy as np
from src.model.cnn import CNNModel
from src.model.loss import CrossEntropyLoss
from src.model.optimizer import AdamOptimizer
from src.trainer.train import Trainer


def test_train():
    """
    Test the Trainer class. Verifies training runs without errors,
    history is correctly returnes, loss values are valid,
    and model weights are updated.
    """

    # Fake data
    X = np.random.randn(5, 1, 28, 28)
    y = np.eye(7)[np.random.randint(0, 7, size=5)]

    model = CNNModel(input_shape=(1, 28, 28))
    loss_fn = CrossEntropyLoss()
    optimizer = AdamOptimizer(lr=0.001)

    # Store initial weights
    initial_weights = []
    for layer in model.layers:
        if hasattr(layer, "weight"):
            initial_weights.append(layer.weight.copy())

    trainer = Trainer(model, loss_fn, optimizer)

    history = trainer.train(X, y, epochs=2)

    # Check history
    assert "loss" in history
    assert "accuracy" in history

    assert len(history["loss"]) == 2
    assert len(history["accuracy"]) == 2

    assert all(loss > 0 for loss in history["loss"])

    # Check weights updated
    updated_weights = []
    for layer in model.layers:
        if hasattr(layer, "weight"):
            updated_weights.append(layer.weight)

    assert any(
        not np.allclose(w0, w1) for w0, w1 in zip(initial_weights, updated_weights)
    )
