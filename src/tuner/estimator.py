import numpy as np

from src.model.cnn import CNNModel
from src.trainer.train import Trainer
from src.trainer.predict import Predictor
from src.evaluation.evaluator import Evaluator
from src.model.loss import CrossEntropyLoss
from src.model.optimizer import AdamOptimizer


class CNNEstimator:
    """
    Sklearn-compatible estimator wrapper for CNNModel.

    This class enables integration with Bayesian optimization
    (e.g., BayesSearchCV) by exposing hyperparameters in a
    standardized interface. Designed for use with skopt.BayesSearchCV
    in mind.
    """

    def __init__(
        self,
        input_shape=(5, 120, 120),
        conv_channels=(8, 16, 32),
        kernel_size=3,
        pool_size=2,
        alpha=0.01,
        dropout_rate=0.15,
        hidden_size=128,
        learning_rate=0.001,
        epochs=5,
        batch_size=32,
    ):
        """
        Initialize the CNNEstimator with hyperparameters.

        :param input_shape: Shape of input tensor image (C, H, W)
        :type input_shape: tuple
        :param conv_channels: Convolution layer channels
        :type conv_channels: tuple
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
        """

        # Model params
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        # Training params
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Internal objects
        self.model = None
        self.trainer = None
        self.predictor = None

    def _build_model(self) -> None:
        """
        Construct the CNN model (loss + optimizer as well),
        trainer, predictor, and evaluator.

        :return: None
        :rtype: None
        """

        self.model = CNNModel(
            input_shape=self.input_shape,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            alpha=self.alpha,
            dropout_rate=self.dropout_rate,
            hidden_size=self.hidden_size,
        )

        loss_fn = CrossEntropyLoss()
        optimizer = AdamOptimizer(lr=self.learning_rate)

        self.trainer = Trainer(self.model, loss_fn, optimizer)
        self.predictor = Predictor(self.model)
        self.evaluator = Evaluator()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNEstimator":
        """
        Train the CNN model.

        :param X: Training data
        :type X: np.ndarray
        :param y: Training labels (one-hot vectors)
        :type y: np.ndarray
        :return: Fitted estimator
        :rtype: CNNEstimator
        """

        self._build_model()

        self.trainer.train(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.

        :param X: Input data
        :type X: np.ndarray
        :return: Predicted class probabilities
        :rtype: np.ndarray
        """

        return self.predictor.predict(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        :param X: Input data
        :type X: np.ndarray
        :return: Predicted class indices
        :rtype: np.ndarray
        """

        return self.predictor.predict_classes(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the model performance using Evaluator.

        :param X: Input data
        :type X: np.ndarray
        :param y: Ground truth labels (one-hot vectors)
        :type y: np.ndarray
        :return: Accuracy score
        :rtype: float
        """

        # Convert one-hot to class indices if needed
        if y.ndim == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y

        y_pred = self.predict_classes(X)

        # Evaluate using Evaluator
        self.evaluator.confusion_matrix(y_true, y_pred)
        metrics = self.evaluator.compute_metrics()

        return metrics["f1_score"]
