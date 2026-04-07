import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
from src.evaluation.evaluator import Evaluator


@pytest.fixture(autouse=True)
def close_plots():
    """
    Automatically close all matplotlib figures after every test
    in this module. Prevents figure accumulation in headless CI environments
    which can corrupt the coverage report.
    """
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def test_confusion_matrix():
    """
    Test the computation of the confusion matrix
    metrics of the Evaluator.

    Vertify that TN, FP, FN, TP are correctly computed
    for binary classification.
    """

    # Initialize the evaluator
    evaluator = Evaluator()

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cm = evaluator.confusion_matrix(y_true, y_pred)

    # Expected values: TN=1, FP=1, FN=1, TP=1
    expected_cm = np.array([[1, 1], [1, 1]])
    assert np.array_equal(cm, expected_cm)


def test_compute_metrics():
    """
    Test the computation of the evaluation metrics
    of the Evaluator.

    Vertify that accuracy, precision, recall, fallout, and
    F1 score are computed correctly for a known confusion matrix,
    and that an error is raised if confusion matrix has not been computed.
    """

    evaluator = Evaluator()

    # Case 1: Confusion matrix exists
    evaluator.cm = np.array([[50, 10], [5, 35]])  # [TN, FP], [FN, TP]

    metrics = evaluator.compute_metrics()

    # Check that all expected metrics exist
    for key in ["accuracy", "precision", "recall", "fallout", "f1_score"]:
        assert key in metrics

    # Accuracy
    assert np.isclose(metrics["accuracy"], 0.85)
    # Precision
    assert np.isclose(metrics["precision"], 7 / 9)
    # Recall
    assert np.isclose(metrics["recall"], 0.875)
    # Fallout
    assert np.isclose(metrics["fallout"], 1 / 6)
    # F1 score
    assert np.isclose(metrics["f1_score"], 0.8235294)

    # Case 2: Confusion matrix has NOT been computed
    evaluator.cm = None  # Reset state
    error_raised = False
    try:
        evaluator.compute_metrics()
    except ValueError as e:
        error_raised = True
        assert str(e) == "Confusion matrix has not been computed."
    assert error_raised, "Expected ValueError when confusion matrix is None"


def test_find_best_threshold():
    """
    Test the threshold determination for the ROC
    curve of the Evaluator.

    Vertify that the method returns a float threshold.
    """

    evaluator = Evaluator()

    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.9, 0.4, 0.8])

    best_thresh = evaluator.find_best_threshold(y_true, probs)

    assert isinstance(best_thresh, float)
    assert 0.0 <= best_thresh <= 1.0


def test_roc_curve():
    """
    Test the computation of the ROC curve
    metrics of the Evaluator.

    Vertify that FPR and TPR
    arrays are computed and have the correct length.
    """

    evaluator = Evaluator()

    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.9, 0.4, 0.8])

    fpr, tpr = evaluator.roc_curve(y_true, probs)

    # Should return arrays of same length
    assert isinstance(fpr, np.ndarray)
    assert isinstance(tpr, np.ndarray)
    assert len(fpr) == len(tpr)
    assert len(fpr) > 1  # at least more than 1 threshold


def test_auc():
    """
    Test the AUC metric of the Evaluator.

    Vertify that the AUC is between 0 and 1.
    """

    evaluator = Evaluator()

    fpr = np.array([0.0, 0.5, 1.0])
    tpr = np.array([0.0, 0.8, 1.0])

    auc_value = evaluator.auc(fpr, tpr)

    assert isinstance(auc_value, float)
    assert 0.0 <= auc_value <= 1.0


def test_precision_recall_curve():
    """
    Test the precision-recall curve metrics
    of the Evaluator.

    Vertifies that the precision and
    recall arrays are computed and have the same length.
    """

    evaluator = Evaluator()

    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.9, 0.4, 0.8])

    recall, precision = evaluator.precision_recall_curve(y_true, probs)

    assert isinstance(recall, np.ndarray)
    assert isinstance(precision, np.ndarray)
    assert len(recall) == len(precision)
    assert len(recall) > 1  # at least more than 1 threshold


def test_update_history():
    """
    Test the training history metrics
    are being updated per epoch of the Evaluator.

    Vertify that the training loss, validation loss, and accuracy
    values are correctly appended to the Evaluator's internal lists
    for epoch-level tracking.
    """

    evaluator = Evaluator()

    # Initial sanity check: lists should be empty
    assert evaluator.train_losses == [], "Expected empty train_losses list initially"
    assert evaluator.val_losses == [], "Expected empty val_losses list initially"
    assert evaluator.accuracies == [], "Expected empty accuracies list initially"

    # Update history for first epoch
    train_loss = 0.5
    val_loss = 0.6
    accuracy = 80.0

    evaluator.update_history(train_loss, val_loss, accuracy)

    # Verify that the values are correctly appended
    assert evaluator.train_losses == [train_loss], "train_losses not updated correctly"
    assert evaluator.val_losses == [val_loss], "val_losses not updated correctly"
    assert evaluator.accuracies == [accuracy], "accuracies not updated correctly"

    # Update history for second epoch
    evaluator.update_history(0.4, 0.55, 85.0)

    # Verify that multiple entries are correctly tracked
    assert evaluator.train_losses == [
        0.5,
        0.4,
    ], "train_losses not tracking multiple epochs"
    assert evaluator.val_losses == [
        0.6,
        0.55,
    ], "val_losses not tracking multiple epochs"
    assert evaluator.accuracies == [
        80.0,
        85.0,
    ], "accuracies not tracking multiple epochs"


def test_plot_confusion_matrix(mocker):
    """
    Test the confusion matrix output of the Evaluator.

    Verifies that the function runs without errors for a valid confusion matrix
    (show=False), and plt.show() is called when show=True.

    :param mocker: pytest mock fixture
    :type mocker: object
    """

    evaluator = Evaluator()
    cm = np.array([[5, 2], [1, 7]])  # Sample confusion matrix

    # Case 1: No GUI
    evaluator.plot_confusion_matrix(cm, show=False)

    # Case 2: Mock plt.show so no GUI is opened
    mock_show = mocker.patch("matplotlib.figure.Figure.show")
    evaluator.plot_confusion_matrix(cm, show=True)

    mock_show.assert_called_once()


def test_plot_roc(mocker):
    """
    Test the the ROC curve of the Evaluator.

    Vertify that the ROC curve plotting runs without errors
    (show=False), and plt.show() is called when show=True.

    :param mocker: pytest mock fixture
    :type mocker: object
    """

    evaluator = Evaluator()
    fpr = np.array([0.0, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.6, 0.8, 1.0])

    # Case 1: No GUI
    evaluator.plot_roc(fpr, tpr, show=False)

    # Case 2: Mock plt.show so no GUI is opened
    mock_show = mocker.patch("matplotlib.figure.Figure.show")
    evaluator.plot_roc(fpr, tpr, show=True)

    mock_show.assert_called_once()


def test_plot_pr(mocker):
    """
    Test the precision-recall curve output
    of the Evaluator.

    Vertify that the precision-recall curve plotting runs without errors
    (show=False), and plt.show() is called when show=True.

    :param mocker: pytest mock fixture
    :type mocker: object
    """

    evaluator = Evaluator()
    recall = np.array([0.0, 0.4, 0.7, 1.0])
    precision = np.array([1.0, 0.8, 0.6, 0.0])

    # Case 1: No GUI
    evaluator.plot_pr(recall, precision, show=False)

    # Case 2: Mock plt.show so no GUI is opened
    mock_show = mocker.patch("matplotlib.figure.Figure.show")
    evaluator.plot_pr(recall, precision, show=True)

    mock_show.assert_called_once()


def test_plot_accuracy(mocker):
    """
    Test the accuracy vs epochs curve output
    of the Evaluator.

    Vertify that the accuracy vs epochs plotting runs without errors
    (show=False), and plt.show() is called when show=True.

    :param mocker: pytest mock fixture
    :type mocker: object
    """

    evaluator = Evaluator()
    history = {"accuracy": [50, 60, 70, 80, 90]}

    # Case 1: No GUI
    evaluator.plot_accuracy(history, show=False)

    # Case 2: Mock plt.show so no GUI is opened
    mock_show = mocker.patch("matplotlib.figure.Figure.show")
    evaluator.plot_accuracy(history, show=True)

    mock_show.assert_called_once()


def test_plot_loss_batches(mocker):
    """
    Test the loss vs training batches curve
    output of the Evaluator.

    Vertify that the loss vs batch plotting runs without errors
    (show=False), and plt.show() is called when show=True.

    :param mocker: pytest mock fixture
    :type mocker: object
    """

    evaluator = Evaluator()
    history = {"batch_loss": [0.9, 0.7, 0.6, 0.5, 0.4]}

    # Case 1: No GUI
    evaluator.plot_loss_batches(history, show=False)

    # Case 2: Mock plt.show so no GUI is opened
    mock_show = mocker.patch("matplotlib.figure.Figure.show")
    evaluator.plot_loss_batches(history, show=True)

    mock_show.assert_called_once()
