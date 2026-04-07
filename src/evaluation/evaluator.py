import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluator class for binary classification performance analysis.

    This class computes scalar evaluation metrics and provides
    visualization utilities such as confusion matrix, ROC curve,
    precision-recall curve, and training diagnostics.
    """

    def __init__(self):
        """
        Initialize evaluator storage.

        :return: None
        :rtype: None
        """

        # Confusion matrix: [[TN, FP], [FN, TP]]
        self.cm: np.ndarray | None = None

        # Dictionary to store computed scalar metrics
        self.metrics: dict = {}

        # Epoch-level metrics (for update_history)
        self.train_losses: list[float] = []  # Training loss per epoch
        self.val_losses: list[float] = []  # Validation loss per epoch
        self.accuracies: list[float] = []  # Accuracy per epoch

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the binary confusion matrix.

        :param y_true: Ground truth labels (0 or 1)
        :type y_true: np.ndarray
        :param y_pred: Predicted labels (0 or 1)
        :type y_pred: np.ndarray
        :return: Confusion matrix [[TN, FP], [FN, TP]]
        :rtype: np.ndarray
        """

        TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives

        # Store confusion matrix
        self.cm = np.array([[TN, FP], [FN, TP]])

        return self.cm

    def compute_metrics(self) -> dict:
        """
        Compute scalar evaluation metrics from the confusion matrix.

        :return: Dictionary containing accuracy, precision, recall,
                fallout, and F1-score
        :rtype: dict
        """

        # Sanity check: Ensure confusion matrix exists
        if self.cm is None:
            raise ValueError("Confusion matrix has not been computed.")

        TN, FP, FN, TP = self.cm.ravel()

        # Compute scalar metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        fallout = FP / (FP + TN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Store metrics
        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fallout": fallout,
            "f1_score": f1,
        }

        return self.metrics

    def find_best_threshold(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """
        Find the optimal classification threshold that maximizes F1 score.

        :param y_true: Ground truth labels (0 or 1)
        :type y_true: np.ndarray
        :param probs: Predicted probabilities for positive class
        :type probs: np.ndarray
        :return: Optimal threshold
        :rtype: float
        """

        thresholds = np.linspace(0.0, 1.0, 100)
        best_f1 = -1
        best_threshold = 0.5  # Default decision threshold

        for t in thresholds:
            # Convert probabilities to binary predictions
            y_pred = (probs >= t).astype(int)

            # Compute metrics at this threshold
            self.confusion_matrix(y_true, y_pred)
            metrics = self.compute_metrics()

            # Track best threshold based on F1 score
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_threshold = t

        return best_threshold

    def roc_curve(self, y_true: np.ndarray, probs: np.ndarray) -> tuple:
        """
        Compute the ROC curve values.

        :param y_true: Ground truth labels (0 or 1)
        :type y_true: np.ndarray
        :param probs: Predicted probabilities
        :type probs: np.ndarray
        :return: Tuple of (FPR array, TPR array)
        :rtype: tuple (np.ndarray, np.ndarray)
        """

        thresholds = np.linspace(0.0, 1.0, 100)

        fpr_list = []
        tpr_list = []

        for t in thresholds:
            # Apply threshold
            y_pred = (probs >= t).astype(int)

            # Compute confusion matrix
            self.confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = self.cm.ravel()

            # Compute rates
            tpr = TP / (TP + FN + 1e-8)  # True Positive Rate
            fpr = FP / (FP + TN + 1e-8)  # False Positive Rate

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(fpr_list), np.array(tpr_list)

    def auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Compute the Area Under the ROC Curve (AUC).

        :param fpr: False Positive Rate values
        :type fpr: np.ndarray
        :param tpr: True Positive Rate values
        :type tpr: np.ndarray
        :return: AUC score
        :rtype: float
        """

        # Trapezoidal rule for numerical integration
        return np.trapezoid(tpr, fpr)

    def precision_recall_curve(self, y_true: np.ndarray, probs: np.ndarray) -> tuple:
        """
        Compute the precision-recall curve.

        :param y_true: Ground truth labels (0 or 1)
        :type y_true: np.ndarray
        :param probs: Predicted probabilities
        :type probs: np.ndarray
        :return: Tuple of (recall array, precision array)
        :rtype: tuple (np.ndarray, np.ndarray)
        """

        thresholds = np.linspace(0.0, 1.0, 100)

        precision_list = []
        recall_list = []

        for t in thresholds:
            y_pred = (probs >= t).astype(int)

            # Compute metrics
            self.confusion_matrix(y_true, y_pred)
            metrics = self.compute_metrics()

            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])

        return np.array(recall_list), np.array(precision_list)

    def update_history(
        self, train_loss: float, val_loss: float, accuracy: float
    ) -> None:
        """
        Update training history (epoch-level).

        :param train_loss: Training loss for current epoch
        :type train_loss: float
        :param val_loss: Validation loss for current epoch
        :type val_loss: float
        :param accuracy: Validation accuracy for current epoch
        :type accuracy: float
        """

        # Store epoch-level metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracies.append(accuracy)

    def plot_confusion_matrix(self, cm: np.ndarray, show: bool = True) -> None:
        """
        Plot the confusion matrix.

        :param cm: Confusion matrix
        :type cm: np.ndarray
        :param show: Whether to display the plot
        :type show: bool
        :return: None
        :rtype: None
        """

        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_roc(self, fpr: np.ndarray, tpr: np.ndarray, show: bool = True) -> None:
        """
        Plot the ROC curve.

        :param fpr: False Positive Rate values
        :type fpr: np.ndarray
        :param tpr: True Positive Rate values
        :type tpr: np.ndarray
        :param show: Whether to display the plot
        :type show: bool
        :return: None
        :rtype: None
        """

        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_pr(
        self, recall: np.ndarray, precision: np.ndarray, show: bool = True
    ) -> None:
        """
        Plot the precision-recall curve.

        :param recall: Recall values
        :type recall: np.ndarray
        :param precision: Precision values
        :type precision: np.ndarray
        :param show: Whether to display the plot
        :type show: bool
        :return: None
        :rtype: None
        """

        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_accuracy(self, history: dict, show: bool = True) -> None:
        """
        Plot the accuracy vs epochs curve.

        :param history: Training history dictionary
        :type history: dict
        :param show: Whether to display the plot
        :type show: bool
        :return: None
        :rtype: None
        """

        plt.figure()
        plt.plot(history["accuracy"])
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_loss_batches(self, history: dict, show: bool = True) -> None:
        """
        Plot the loss vs training batches curve.

        :param history: Training history dictionary
        :type history: dict
        :param show: Whether to display the plot
        :type show: bool
        :return: None
        :rtype: None
        """

        plt.figure()
        plt.plot(history["batch_loss"])
        plt.title("Loss vs Training Batches")
        plt.xlabel("Batch")
        plt.ylabel("Loss")

        if show:
            plt.show()
        else:
            plt.close()
