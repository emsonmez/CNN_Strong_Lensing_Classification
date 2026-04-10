import numpy as np


class Trainer:
    """
    Trainer class for handling the training loop of the CNN model.

    This class performs mini-batch gradient descent and tracks both
    batch-level and epoch-level statistics for downstream evaluation.
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

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> dict:
        """
        Train the model using mini-batch gradient descent.

        :param X: Input data
        :type X: np.ndarray
        :param y: Ground truth labels (one-hot encoded)
        :type y: np.ndarray
        :param X_val: Validation input data
        :type X_val: np.ndarray
        :param y_val: Validation labels
        :type y_val: np.ndarray
        :param epochs: Number of training epochs
        :type epochs: int
        :param batch_size: Number of samples per batch
        :type batch_size: int
        :return: Training history containing batch-level and epoch-level metrics
        :rtype: dict
        """

        # Making sure all downstream operations is consistent
        # regardless of how the labels were originally supplied
        num_classes = self.model.forward(X[:1], training=False).shape[1]
        if y.ndim == 1:
            y = np.eye(num_classes)[y.astype(int)]
        if y_val is not None and y_val.ndim == 1:
            y_val = np.eye(num_classes)[y_val.astype(int)]

        num_samples = len(X)

        # History dictionary for tracking metrics
        history = {
            "epoch_loss": [],  # Average loss per epoch
            "epoch_accuracy": [],  # Accuracy per epoch
            "batch_loss": [],  # Loss per batch (for plotting)
            "val_loss": [],  # Validation loss per epoch
            "val_accuracy": [],  # Validation accuracy per epoch
        }

        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0

            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Iterate over mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass (batch)
                y_pred = self.model.forward(X_batch, training=True)

                # Compute batch loss
                loss = self.loss_fn.forward(y_pred, y_batch)

                # Store batch loss (enables loss vs batches plot)
                history["batch_loss"].append(loss)

                # Accumulate weighted loss for epoch average
                total_loss += loss * len(X_batch)

                # Accuracy (batch; guaranteed 2D here)
                correct_predictions += np.sum(
                    np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1)
                )

                # Backward pass (lr = 0; optimizer handles updates)
                dL = self.loss_fn.backward()
                self.model.backward(dL, lr=0)

                # Optimizer step
                self.optimizer.step(self.model.layers)

            # Epoch metrics
            avg_loss = total_loss / num_samples
            accuracy = (correct_predictions / num_samples) * 100.0

            history["epoch_loss"].append(avg_loss)
            history["epoch_accuracy"].append(accuracy)

            if X_val is not None and y_val is not None:
                # Forward pass (inference mode)
                y_val_pred = self.model.forward(X_val, training=False)

                # Validation loss
                val_loss = self.loss_fn.forward(y_val_pred, y_val)

                # Validation accuracy (guaranteed 2D here)
                val_accuracy = (
                    np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                    * 100.0
                )

                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {avg_loss:.4f} "
                    f"- Acc: {accuracy:.2f}% "
                    f"- Val Loss: {val_loss:.4f} "
                    f"- Val Acc: {val_accuracy:.2f}%"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {avg_loss:.4f} "
                    f"- Accuracy: {accuracy:.2f}%"
                )

        return history
