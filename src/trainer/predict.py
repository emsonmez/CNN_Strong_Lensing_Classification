import numpy as np


class Predictor:
    """
    Predictor class for handling inference using the trained CNN model.

    This class provides methods to generate predictions and class labels
    from input data by performing forward passes through the model
    in inference mode. Generalized for both single-image and batch inputs.
    """

    def __init__(self, model):
        """
        Initialize the predictor.

        :param model: Trained CNN model
        :type model: object
        """

        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.

        This method performs a forward pass through the model with
        training-specific behavior disabled.

        :param X: Input data (single sample or batch)
        :type X: np.ndarray
        :return: Model output predictions (logits or probabilities)
        :rtype: np.ndarray
        """

        # Ensure input has batch dimension
        if X.ndim == 3:
            X = X[np.newaxis, ...]

        # Forward pass in inference mode
        predictions = self.model.forward(X, training=False)

        return predictions

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        This method converts model outputs into discrete class predictions.

        :param X: Input data (single sample or batch)
        :type X: np.ndarray
        :return: Predicted class indices
        :rtype: np.ndarray
        """

        predictions = self.predict(X)

        # Convert to class labels
        class_labels = np.argmax(predictions, axis=1)

        return class_labels
