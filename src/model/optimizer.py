import numpy as np
from typing import Optional


class AdamOptimizer:
    """
    Adaptive Moment Estimation (ADAM) optimizer.

    Uses a stochastic adaptive moment estimation 
    for parameter updates.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize ADAM parameters according to 
        the proposed values in Kingma & Ba (2015): 
        DOI: 10.48550/arXiv.1412.6980.

        :param lr: Learning rate (α)
        :type lr: float
        :param beta1: First moment exponential decay rate
        :type beta1: float
        :param beta2: Second moment exponential decay rate
        :type beta2: float
        :param epsilon: Numerical stability term
        :type epsilon: float
        """

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Timestep and (First and Second) Moment Vectors
        self.m = {}
        self.v = {}
        self.t = 0


    def step(self, layers: list) -> None:
        """
        Compute the ADAM update step.

        :param layers: List of model layers
        :type layers: list
        """

        self.t += 1

        for idx, layer in enumerate(layers):

            if not hasattr(layer, "dL_dweight"):
                continue

            # Initialize moments
            if idx not in self.m:
                self.m[idx] = {
                    "w": np.zeros_like(layer.weight),
                    "b": np.zeros_like(layer.bias)
                }
                self.v[idx] = {
                    "w": np.zeros_like(layer.weight),
                    "b": np.zeros_like(layer.bias)
                }

            # Gradients
            g_w = layer.dL_dweight
            g_b = layer.dL_dbias

            # First moment
            self.m[idx]["w"] = self.beta1 * self.m[idx]["w"] + (1 - self.beta1) * g_w
            self.m[idx]["b"] = self.beta1 * self.m[idx]["b"] + (1 - self.beta1) * g_b

            # Second moment
            self.v[idx]["w"] = self.beta2 * self.v[idx]["w"] + (1 - self.beta2) * (g_w ** 2)
            self.v[idx]["b"] = self.beta2 * self.v[idx]["b"] + (1 - self.beta2) * (g_b ** 2)

            # Bias correction
            m_hat_w = self.m[idx]["w"] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[idx]["w"] / (1 - self.beta2 ** self.t)

            m_hat_b = self.m[idx]["b"] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v[idx]["b"] / (1 - self.beta2 ** self.t)

            # Update rule
            layer.weight -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)