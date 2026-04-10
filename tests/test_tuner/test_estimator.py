import numpy as np
from src.tuner.estimator import CNNEstimator


def test_fit():
    """
    Test the fit function of the CNNEstimator.

    Vertify that there are no runtime errors,
    model is trained, and the estimator returns itself
    confirming that it's sklearn compatible.
    """

    X = np.random.randn(6, 5, 28, 28)
    y = np.eye(2)[np.random.randint(0, 2, size=6)]

    estimator = CNNEstimator(input_shape=(5, 28, 28), epochs=2, batch_size=3)

    result = estimator.fit(X, y)

    assert result is estimator


def test_predict():
    """
    Test the predict function of the CNNEstimator.

    Vertify that the output shape is (N, num_classes),
    and the values are finite.
    """

    X = np.random.randn(4, 5, 28, 28)
    y = np.eye(2)[np.random.randint(0, 2, size=4)]

    estimator = CNNEstimator(input_shape=(5, 28, 28), epochs=1)
    estimator.fit(X, y)

    preds = estimator.predict(X)

    assert preds.shape == (4, 2)
    assert np.all(np.isfinite(preds))


def test_predict_classes():
    """
    Test the class predictions function of the CNNEstimator.

    Vertify that the output shape is (N,),
    values are valid class indices, and
    that it matches argmax of predict().
    """

    X = np.random.randn(5, 5, 28, 28)
    y = np.eye(2)[np.random.randint(0, 2, size=5)]

    estimator = CNNEstimator(input_shape=(5, 28, 28), epochs=1)
    estimator.fit(X, y)

    preds = estimator.predict(X)
    classes = estimator.predict_classes(X)

    assert classes.shape == (5,)
    assert np.all(classes >= 0)
    assert np.all(classes < 2)

    assert np.all(classes == np.argmax(preds, axis=1))


def test_score():
    """
    Test the scoring function of the CNNEstimator.

    Vertify that the score returns a float, and
    the score is within the valid range of [0, 1]
    for both one-hot encoded labels (2D) and integer class
    labels (1D).
    """

    estimator = CNNEstimator(input_shape=(5, 28, 28), epochs=2)
    X = np.random.randn(6, 5, 28, 28)

    # Case 1: y is a 2D one-hot encoded label
    y_onehot = np.eye(2)[np.random.randint(0, 2, size=6)]
    estimator.fit(X, y_onehot)
    score_onehot = estimator.score(X, y_onehot)
    assert isinstance(score_onehot, float)
    assert 0.0 <= score_onehot <= 1.0

    # Case 2: y is a 1D integer class indices label
    y_labels = np.random.randint(0, 2, size=6)
    estimator.fit(X, y_labels)
    score_labels = estimator.score(X, y_labels)
    assert isinstance(score_labels, float)
    assert 0.0 <= score_labels <= 1.0
