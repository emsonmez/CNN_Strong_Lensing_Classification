import numpy as np
from src.model.cnn import CNNModel
from src.trainer.predict import Predictor


def test_predict():
    """
    Test the predict function.

    Verify predictions run without errors,
    output shapes are correct for both single-image
    and batch inputs, and values are finite.
    """

    model = CNNModel(input_shape=(1, 120, 120))
    predictor = Predictor(model)

    # ---- Single image ----
    X_single = np.random.randn(1, 120, 120)
    predictions_single = predictor.predict(X_single)

    # Should automatically add batch dimension
    assert predictions_single.shape == (1, 7)

    # Check values are valid
    assert np.all(np.isfinite(predictions_single))

    # ---- Batch input ----
    X_batch = np.random.randn(3, 1, 120, 120)
    predictions_batch = predictor.predict(X_batch)

    assert predictions_batch.shape == (3, 7)

    assert np.all(np.isfinite(predictions_batch))


def test_predict_classes():
    """
    Test the predict_classes function.

    Verifies class predictions
    are correct shape, within valid range, and consistent with
    raw predict outputs.
    """

    model = CNNModel(input_shape=(1, 120, 120))
    predictor = Predictor(model)

    # Fake data
    X = np.random.randn(4, 1, 120, 120)

    predictions = predictor.predict(X)
    class_preds = predictor.predict_classes(X)

    assert class_preds.shape == (4,)

    # Check valid class indices
    assert np.all(class_preds >= 0)
    assert np.all(class_preds < 7)

    # Check consistency with argmax
    assert np.all(class_preds == np.argmax(predictions, axis=1))
