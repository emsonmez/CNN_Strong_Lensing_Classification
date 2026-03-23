import numpy as np


class Trainer:
    """
    Trainer class for handling the training loop of the CNN model.
    """

    def __init__(self, model, loss_fn, optimizer):
        """
        Initialize the trainer models.

        :param model: CNN model
        :type model: object
        :param loss_fn: Loss function
        :type loss_fn: object
        :param optimizer: Optimizer (e.g., Adam)
        :type optimizer: object
        """

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> dict:
        """
        Train the model.

        :param X: Input data
        :type X: np.ndarray
        :param y: Ground truth labels (one-hot encoded)
        :type y: np.ndarray
        :param epochs: Number of training epochs
        :type epochs: int
        :return: Training history (loss and accuracy)
        :rtype: dict
        """

        num_samples = len(X)

        history = {
            "loss": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0

            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(num_samples):
                x_sample = X_shuffled[i][np.newaxis, ...]
                y_sample = y_shuffled[i][np.newaxis, ...]

                # Forward pass
                y_pred = self.model.forward(x_sample, training=True)

                loss = self.loss_fn.forward(y_pred, y_sample)
                total_loss += loss

                # Accuracy
                if np.argmax(y_pred) == np.argmax(y_sample):
                    correct_predictions += 1

                # Backward pass
                dL = self.loss_fn.backward()

                # lr = 0 (optimizer handles updates)
                self.model.backward(dL, lr=0)

                # Optimizer step
                self.optimizer.step(self.model.layers)

            # Epoch metrics
            avg_loss = total_loss / num_samples
            accuracy = (correct_predictions / num_samples) * 100.0

            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {avg_loss:.4f} "
                f"- Accuracy: {accuracy:.2f}%"
            )

        return history
