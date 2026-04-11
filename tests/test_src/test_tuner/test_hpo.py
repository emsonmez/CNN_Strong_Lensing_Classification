import numpy as np
import pytest
from skopt.space import Categorical

from src.tuner.hpo import Tuner
from src.tuner.estimator import CNNEstimator


def test_build_search_space():
    """
    Test that the search space function of the Tuner.

    Vertify that the required hyperparameters exist, including
    conv_channels is explicitly encoded as comma-separated strings.
    """

    tuner = Tuner()
    space = tuner._build_search_space()

    expected_keys = {
        "conv1_channels",
        "conv2_channels",
        "conv3_channels",
        "kernel_size",
        "pool_size",
        "alpha",
        "hidden_size",
        "dropout_rate",
        "learning_rate",
        "epochs",
        "batch_size",
    }

    assert set(space.keys()) == expected_keys

    # conv_channels as a tuple must not appear
    assert "conv_channels" not in space

    # Each channel dimension must contain only plain scalar integers
    for key in ("conv1_channels", "conv2_channels", "conv3_channels"):
        assert isinstance(space[key], Categorical)
        assert all(isinstance(v, int) for v in space[key].categories)


def test_tune():
    """
    Test the Bayesian tune optimization function of the Tuner.

    Vertify that there are no runtime errors,
    the search object is created,
    and the best params and score are stored.
    Generalized for both one-hot and integer labels.
    """

    X = np.random.randn(10, 5, 28, 28)

    # Case 1: Integer Labels
    y_int = np.random.randint(0, 2, size=10)

    tuner_int = Tuner(
        input_shape=(5, 28, 28),
        n_iter=2,  # keeping it small for the CI
        cv=2,
    )

    search_int = tuner_int.tune(X, y_int)

    assert search_int is not None
    assert tuner_int.search_ is not None

    assert tuner_int.best_params_ is not None
    assert isinstance(tuner_int.best_params_, dict)

    assert tuner_int.best_score_ is not None
    assert isinstance(tuner_int.best_score_, float)

    assert search_int is not None

    # Case 2: One-hot labels
    y_onehot = np.eye(2)[y_int]

    tuner_oh = Tuner(
        input_shape=(5, 28, 28),
        n_iter=2,
        cv=2,
    )

    search_oh = tuner_oh.tune(X, y_onehot)

    assert tuner_oh.best_params_ is not None
    assert isinstance(tuner_oh.best_params_, dict)

    assert tuner_oh.best_score_ is not None
    assert isinstance(tuner_oh.best_score_, float)

    assert search_oh is not None


def test_get_best_model():
    """
    Test the best model get received function of the Tuner.

    Vertify that the returned object is "CNNEstimator."
    """

    X = np.random.randn(10, 5, 28, 28)
    y = np.random.randint(0, 2, size=10)

    tuner = Tuner(
        input_shape=(5, 28, 28),
        n_iter=2,
        cv=2,
    )

    with pytest.raises(ValueError):
        tuner.get_best_model()

    tuner.tune(X, y)

    model = tuner.get_best_model()

    assert isinstance(model, CNNEstimator)


def test_summary():
    """
    Test the summary output function of the Tuner.

    Vertify that the output contains best_params and best_f1_score
    and for edge cases.
    """

    X = np.random.randn(10, 5, 28, 28)
    y = np.random.randint(0, 2, size=10)

    tuner = Tuner(
        input_shape=(5, 28, 28),
        n_iter=2,
        cv=2,
    )

    # summary raises error if called before tuning
    with pytest.raises(ValueError):
        tuner.summary()

    # get_best_model raises error if called before tuning
    with pytest.raises(ValueError):
        tuner.get_best_model()
    tuner.tune(X, y)

    summary = tuner.summary()

    assert "best_params" in summary
    assert "best_f1_score" in summary

    assert isinstance(summary["best_params"], dict)
    assert isinstance(summary["best_f1_score"], float)
