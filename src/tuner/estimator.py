import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.model.cnn import CNNModel
from src.trainer.train import Trainer
from src.trainer.predict import Predictor
from src.evaluation.evaluator import Evaluator
from src.model.loss import CrossEntropyLoss
from src.model.optimizer import AdamOptimizer


class CNNEstimator(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible estimator wrapper for CNNModel.

    This class enables integration with Bayesian optimization
    (e.g., BayesSearchCV) by exposing hyperparameters in a
    standardized interface.
    """

    # Hyperparameter names explicit to prevent objects
    # from leaking into the cloning mechanism
    _HPARAM_KEYS = (
        "input_shape",
        "conv1_channels",
        "conv2_channels",
        "conv3_channels",
        "kernel_size",
        "pool_size",
        "alpha",
        "dropout_rate",
        "hidden_size",
        "learning_rate",
        "epochs",
        "batch_size",
        "patience",
        "min_delta",
        "random_state",
    )

    def __init__(
        self,
        input_shape: tuple = (5, 120, 120),
        conv1_channels: int = 8,
        conv2_channels: int = 16,
        conv3_channels: int = 32,
        kernel_size: int = 3,
        pool_size: int = 2,
        alpha: float = 0.01,
        dropout_rate: float = 0.15,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = 32,
        patience: int = 5,
        min_delta: float = 1e-4,
        random_state: int = 917,
    ):
        """
        Initialize the CNNEstimator with hyperparameters.

        :param input_shape: Shape of input tensor image (C, H, W)
        :type input_shape: tuple
        :param conv1_channels: Output channels for first conv block.
        :type conv1_channels: int
        :param conv2_channels: Output channels for second conv block.
        :type conv2_channels: int
        :param conv3_channels: Output channels for third conv block.
        :type conv3_channels: int
        :param kernel_size: Convolution kernel size
        :type kernel_size: int
        :param pool_size: Max pooling size (Note that pool_size = stride)
        :type pool_size: int
        :param alpha: LeakyReLU Activation slope
        :type alpha: float
        :param dropout_rate: Dropout rate
        :type dropout_rate: float
        :param hidden_size: Dense layer size
        :type hidden_size: int
        :param learning_rate: Optimizer learning rate
        :type learning_rate: float
        :param epochs: Training epochs
        :type epochs: int
        :param batch_size: Batch size
        :type batch_size: int
        :param patience: Early stopping patience
        :type patience: int
        :param min_delta: Minimum improvement threshold
        :type min_delta: float
        :param random_state: Random seed
        :type random_state: int
        """

        # Model params
        self.input_shape = input_shape
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.conv3_channels = conv3_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        # Training params
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state

        # Fitted Internal objects (trailing underscores excludes them explicitly)
        self.model_ = None
        self.trainer_ = None
        self.predictor_ = None
        self.evaluator_ = None
        self.loss_fn_ = None

    def get_params(self, deep: bool = True) -> dict:
        """
        Return only the hyperparameters, not the fitted internal objects.

        :param deep: Unused; included for sklearn API compatibility.
        :type deep: bool
        :return: Dictionary of hyperparameter names to current values.
        :rtype: dict
        """

        return {key: getattr(self, key) for key in self._HPARAM_KEYS}

    def set_params(self, **params: dict) -> "CNNEstimator":
        """
        Set hyperparameters and re-normalise types injected by skopt
        into plain Python natives.

        :param params: Hyperparameter names and values to update.
        :type params: dict
        :return: Self, for sklearn chaining compatibility.
        :rtype: CNNEstimator
        """

        for k, v in params.items():
            setattr(self, k, v)

        # re-normalize params after skopt injection
        self.conv1_channels = int(self.conv1_channels)
        self.conv2_channels = int(self.conv2_channels)
        self.conv3_channels = int(self.conv3_channels)
        self.kernel_size = int(self.kernel_size)
        self.pool_size = int(self.pool_size)
        self.hidden_size = int(self.hidden_size)
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.patience = int(self.patience)
        self.alpha = float(self.alpha)
        self.dropout_rate = float(self.dropout_rate)
        self.learning_rate = float(self.learning_rate)
        self.min_delta = float(self.min_delta)

        return self

    def _build_model(self) -> None:
        """
        Construct the model (loss + optimizer as well),
        trainer, predictor, and evaluator.

        :return: None
        :rtype: None
        """

        # Reconstruct conv_channels tuple from the three decomposed params
        conv_channels = (
            int(self.conv1_channels),
            int(self.conv2_channels),
            int(self.conv3_channels),
        )

        self.model_ = CNNModel(
            input_shape=self.input_shape,
            conv_channels=conv_channels,
            kernel_size=int(self.kernel_size),
            pool_size=int(self.pool_size),
            alpha=float(self.alpha),
            dropout_rate=float(self.dropout_rate),
            hidden_size=int(self.hidden_size),
        )

        self.loss_fn_ = CrossEntropyLoss()
        optimizer = AdamOptimizer(lr=float(self.learning_rate))

        self.trainer_ = Trainer(self.model_, self.loss_fn_, optimizer)
        self.predictor_ = Predictor(self.model_)
        self.evaluator_ = Evaluator()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNEstimator":
        """
        Train the CNN model using mini-batch gradient descent with
        early stopping based on validation loss.

        :param X: Training data
        :type X: np.ndarray
        :param y: Training labels (one-hot vectors or integer indices)
        :type y: np.ndarray
        :return: Fitted estimator
        :rtype: CNNEstimator
        """

        np.random.seed(self.random_state)
        self._build_model()

        # Train / validation shuffle and split
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))

        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize early stoppoing state
        best_loss = np.inf
        counter = 0

        for _ in range(int(self.epochs)):
            # Train one epoch
            self.trainer_.train(
                X_train,
                y_train,
                epochs=1,
                batch_size=int(self.batch_size),
            )
            # Validation loss step
            proba = self.predict_proba(X_val)
            val_loss = float(self.loss_fn_.forward(proba, y_val))

            # Early stopping check
            if val_loss < best_loss - float(self.min_delta):
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= int(self.patience):
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        :param X: Input data of shape (N, C, H, W).
        :type X: np.ndarray
        :return: Predicted class index array of shape (N,).
        :rtype: np.ndarray

        """
        return self.predictor_.predict_classes(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.

        :param X: Input data of shape (N, C, H, W)
        :type X: np.ndarray
        :return: Predicted class probabilities of shape (N, num_classes)
        :rtype: np.ndarray
        """

        return self.predictor_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the model performance using Evaluator.

        :param X: Input data
        :type X: np.ndarray
        :param y: Ground truth labels (one-hot vectors or integer indices)
        :type y: np.ndarray
        :return: Accuracy score
        :rtype: float
        """

        # Convert one-hot to class indices if needed
        if y.ndim == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y

        y_pred = self.predict(X)

        # Evaluate using Evaluator
        self.evaluator_.confusion_matrix(y_true, y_pred)
        metrics = self.evaluator_.compute_metrics()

        return float(metrics["f1_score"])
