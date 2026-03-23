import numpy as np


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
        epsilon: float = 1e-8,
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

        self.t += 1  # Time step

        for idx, layer in enumerate(layers):
            # Determine if layer has standard parameters (weight, bias)
            # or parameters needed to be updated, but not used (gamma, beta)
            if hasattr(layer, "dL_dweight") and hasattr(layer, "weight"):
                # Standard layer (Conv/Dense)
                grad_pairs = [("weight", "dL_dweight"), ("bias", "dL_dbias")]
            elif hasattr(layer, "dL_dgamma") and hasattr(layer, "gamma"):
                # BatchNorm layer
                grad_pairs = [("gamma", "dL_dgamma"), ("beta", "dL_dbeta")]
            else:
                continue  # No parameters to update

            # Initialize moments if first time
            if idx not in self.m:
                self.m[idx] = {}
                self.v[idx] = {}
                for param_name, _ in grad_pairs:
                    self.m[idx][param_name] = np.zeros_like(getattr(layer, param_name))
                    self.v[idx][param_name] = np.zeros_like(getattr(layer, param_name))

            # Update each parameter using Adam
            for param_name, grad_name in grad_pairs:
                g = getattr(layer, grad_name)

                # First moment (same general structure)
                self.m[idx][param_name] = (
                    self.beta1 * self.m[idx][param_name] + (1 - self.beta1) * g
                )
                # Second moment(same general structure)
                self.v[idx][param_name] = (
                    self.beta2 * self.v[idx][param_name] + (1 - self.beta2) * g**2
                )
                # Bias-corrected moments
                m_hat = self.m[idx][param_name] / (1 - self.beta1**self.t)
                v_hat = self.v[idx][param_name] / (1 - self.beta2**self.t)
                # Update rule
                setattr(
                    layer,
                    param_name,
                    getattr(layer, param_name)
                    - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon),
                )
