import numpy as np
from src.tuner.estimator import CNNEstimator


def test_get_params():
    """
    Test the get_params function of the CNNEstimator.

    Vertify that it returns only hyperparameter keys,
    excludes all fitted internal objects, and is sklearn clone() compatibility.
    """

    estimator = CNNEstimator(input_shape=(5, 28, 28))
    params = estimator.get_params()

    # Must contain all hyperparameter keys
    for key in CNNEstimator._HPARAM_KEYS:
        assert key in params, f"Missing hyperparameter key: {key}"

    # Must NOT contain fitted internal objects
    for fitted in ("model_", "trainer_", "predictor_", "evaluator_", "loss_fn_"):
        assert fitted not in params, f"Fitted object leaked into get_params: {fitted}"


def test_set_params():
    """
    Test the set_params function of the CNNEstimator.

    Vertify that it correctly updates hyperparameters, re-normalises
    numpy scalar types injected by skopt into plain Python natives,
    and returns self for sklearn chaining.
    """

    estimator = CNNEstimator(input_shape=(5, 28, 28))

    # Case 1: plain Python types; basic update and chaining
    result = estimator.set_params(learning_rate=0.005, hidden_size=256)
    assert estimator.learning_rate == 0.005
    assert estimator.hidden_size == 256
    assert result is estimator, "set_params must return self for chaining"

    # Case 2: numpy scalar types as injected by skopt
    estimator.set_params(
        conv1_channels=np.int64(16),
        conv2_channels=np.int64(32),
        conv3_channels=np.int64(64),
        kernel_size=np.int64(5),
        pool_size=np.int64(2),
        hidden_size=np.int64(128),
        epochs=np.int64(3),
        batch_size=np.int64(16),
        patience=np.int64(3),
        alpha=np.float64(0.02),
        dropout_rate=np.float64(0.3),
        learning_rate=np.float64(1e-3),
        min_delta=np.float64(1e-4),
    )
    assert isinstance(estimator.conv1_channels, int)
    assert isinstance(estimator.conv2_channels, int)
    assert isinstance(estimator.conv3_channels, int)
    assert isinstance(estimator.kernel_size, int)
    assert isinstance(estimator.pool_size, int)
    assert isinstance(estimator.hidden_size, int)
    assert isinstance(estimator.epochs, int)
    assert isinstance(estimator.batch_size, int)
    assert isinstance(estimator.patience, int)
    assert isinstance(estimator.alpha, float)
    assert isinstance(estimator.dropout_rate, float)
    assert isinstance(estimator.learning_rate, float)
    assert isinstance(estimator.min_delta, float)


def test_fit():
    """
    Test the fit function of the CNNEstimator.

    Vertify that there are no runtime errors,
    model is built and trained, the estimator returns itself
    confirming that it's sklearn compatible, and
    checks the early stopping logic executes without failure.
    """

    X = np.random.randn(6, 5, 28, 28)
    y = np.eye(2)[np.random.randint(0, 2, size=6)]

    estimator = CNNEstimator(
        input_shape=(5, 28, 28), epochs=10, batch_size=3, patience=1, min_delta=1e-3
    )

    result = estimator.fit(X, y)

    assert result is estimator

    assert estimator.model_ is not None
    assert estimator.predictor_ is not None

    # Predicitions must work after training
    proba = estimator.predict_proba(X)
    assert proba.shape == (6, 2)


def test_predict():
    """
    Test the predict function of the CNNEstimator.

    Vertify that it returns a integer class labels of shape (N,).
    """

    X = np.random.randn(4, 5, 28, 28)
    y = np.eye(2)[np.random.randint(0, 2, size=4)]

    estimator = CNNEstimator(input_shape=(5, 28, 28), epochs=1)
    estimator.fit(X, y)

    preds = estimator.predict(X)

    assert preds.shape == (4,)
    assert np.all(preds >= 0)
    assert np.all(preds < 2)

    assert np.all(preds == np.argmax(estimator.predict_proba(X), axis=1))


def test_predict_proba():
    """
    Test the probability predictions function of the CNNEstimator.

    Vertify that the output shape is (N, num_classes), and all values
    are finite probabilities.
    """

    X = np.random.randn(4, 5, 28, 28)
    y = np.eye(2)[np.random.randint(0, 2, size=4)]

    estimator = CNNEstimator(input_shape=(5, 28, 28), epochs=1)
    estimator.fit(X, y)

    proba = estimator.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.all(np.isfinite(proba))


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
