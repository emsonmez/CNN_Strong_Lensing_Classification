import numpy as np
from typing import Dict
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import StratifiedKFold

from src.tuner.estimator import CNNEstimator


class Tuner:
    """
    Hyperparameter tuning class using Bayesian optimization.

    This class wraps skopt's BayesSearchCV to optimize CNN hyperparameters
    using a probabilistic surrogate model. The objective function maximizes
    the F1-score computed by the CNNEstimator.
    """

    def __init__(
        self,
        input_shape: tuple = (5, 120, 120),
        n_iter: int = 32,
        cv: int = 3,
        random_state: int = 197,
    ):
        """
        Initialize the Tuner.

        :param input_shape: Shape of input tensor image (C, H, W)
        :type input_shape: tuple
        :param n_iter: Number of Bayesian optimization iterations
        :type n_iter: int
        :param cv: Number of cross-validation folds
        :type cv: int
        :param random_state: Random seed for reproducibility
        :type random_state: int
        """

        self.input_shape = input_shape
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

        # Placeholder for best results
        self.best_params_: Dict = None
        self.best_score_: float = None
        self.search_: BayesSearchCV = None

    def _build_search_space(self) -> dict:
        """
        Define the hyperparameter search space.

        :return: Dictionary defining parameter distributions
        :rtype: dict
        """

        return {
            # CNN architecture
            "conv1_channels": Categorical([8, 16, 32]),
            "conv2_channels": Categorical([16, 32, 64]),
            "conv3_channels": Categorical([64, 128, 256]),
            "kernel_size": Categorical([3]),
            "pool_size": Categorical([2]),
            # Activation
            "alpha": Real(1e-4, 0.1, prior="log-uniform"),
            # Dense + regularization
            "hidden_size": Integer(128, 512),
            "dropout_rate": Real(0.1, 0.5),
            # Optimization
            "learning_rate": Real(1e-4, 1e-2, prior="log-uniform"),
            # Training Dynamics
            "epochs": Categorical([3, 5, 7, 10]),
            "batch_size": Categorical([16, 32, 64]),
        }

    def tune(self, X: np.ndarray, y: np.ndarray) -> BayesSearchCV:
        """
        Run Bayesian hyperparameter optimization.

        :param X: Training data
        :type X: np.ndarray
        :param y: Training labels (one-hot vectors or integer indices)
        :type y: np.ndarray
        :return: Fitted BayesSearchCV object
        :rtype: BayesSearchCV
        """

        # Convert one-hot labels to class indices if needed
        if y.ndim == 2:
            y_cv = np.argmax(y, axis=1)
        else:
            y_cv = y

        # Initialize estimator and search space
        estimator = CNNEstimator(input_shape=self.input_shape)
        search_spaces = self._build_search_space()

        # Cross-validation strategy (preserves class balance)
        cv_strategy = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        # Bayesian optimization
        self.search_ = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=self.n_iter,
            scoring="f1",  # optimize based on F1-score
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
            error_score=0.0,
        )

        # Run search
        self.search_.fit(X, y_cv)

        # Store best results
        self.best_params_ = self.search_.best_params_
        self.best_score_ = self.search_.best_score_

        return self.search_

    def get_best_model(self) -> CNNEstimator:
        """
        Retrieve the best trained model after tuning.

        :return: Best estimator found during search
        :rtype: CNNEstimator
        """

        if self.search_ is None:
            raise ValueError("No search has been performed yet.")

        return self.search_.best_estimator_

    def summary(self) -> dict:
        """
        Return a summary of tuning results.

        :return: Dictionary with best parameters and F1-scores
        :rtype: dict
        """

        if self.best_params_ is None:
            raise ValueError("No tuning results available.")

        return {
            "best_params": self.best_params_,
            "best_f1_score": self.best_score_,
        }
