import numpy as np
from typing import Optional


class CrossEntropyLoss:
    """
    Multiclass/Categorical Cross-Entropy loss for multi-class classification.

    Measures the difference between predicted probabilities
    and true class labels.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize loss parameters.

        :param epsilon: Small value to prevent log(0) --> inf
        :type epsilon: float
        """

        self.epsilon = epsilon

        # Stored for backward pass
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None


    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the forward pass of the cross-entropy loss.

        :param y_pred: Predicted probabilities (after softmax)
        :type y_pred: np.ndarray
        :param y_true: True labels (one-hot encoded)
        :type y_true: np.ndarray
        :return: Scalar loss value
        :rtype: float
        """

        self.y_pred = y_pred
        self.y_true = y_true

        # Number of samples
        N = y_pred.shape[0]

        # Add epsilon for numerical stability
        y_pred_stable = np.clip(self.y_pred, self.epsilon, 1.0 - self.epsilon)

        # Compute cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred_stable)) / N

        return loss

    def backward(self) -> np.ndarray:
        """
        Compute the backward pass of the cross-entropy loss.

        :return: Gradient of the loss with respect to predictions
        :rtype: np.ndarray
        """

        N = self.y_pred.shape[0]

        y_pred_stable = np.clip(self.y_pred, self.epsilon, 1.0 - self.epsilon)

        # Gradient simplifies with softmax
        dL_dz = (y_pred_stable - self.y_true) / N

        return dL_dz