import numpy as np
from src.model.optimizer import AdamOptimizer


def test_step():
    """
    Test the step function of AdamOptimizer.

    Verify that parameters update in the negative gradient direction
    and continue updating correctly across multiple steps.
    Also checks if non-trainable layers are safely ignored.
    """

    # Create dummy layer with simple parameters
    class DummyLayer:
        def __init__(self):
            self.weight = np.array([[1.0]])
            self.bias = np.array([1.0])

            # Constant positive gradients
            self.dL_dweight = np.array([[2.0]])
            self.dL_dbias = np.array([2.0])

    # Non-trainable layer (no gradients)
    class NonTrainableLayer:
        def __init__(self):
            self.some_value = 5

    # Initialize the class with our AdamOptimizer class
    layer = DummyLayer()
    non_trainable = NonTrainableLayer()
    optimizer = AdamOptimizer(lr=0.1)  # Bigger steps then default

    weight_before = layer.weight.copy()
    bias_before = layer.bias.copy()

    # First update step; Should not crash
    optimizer.step([layer, non_trainable])

    # After positive gradient, parameters should decrease
    assert layer.weight < weight_before
    assert layer.bias < bias_before

    # Store values after first update
    weight_after_step1 = layer.weight.copy()
    bias_after_step1 = layer.bias.copy()

    # Second update step (same gradients running through the same code)
    optimizer.step([layer])

    # Parameters should continue decreasing for a positive second order gradient
    assert layer.weight < weight_after_step1
    assert layer.bias < bias_after_step1
