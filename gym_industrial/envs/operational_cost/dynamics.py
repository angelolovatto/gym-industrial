"""Python implementation of OperationalCost state transition."""
from dataclasses import dataclass

import numpy as np


@dataclass
class OperationalCostParams:
    """Parameters for Operational Cost dynamics."""

    setpoint_cost: float = 2.0
    velocity_cost: float = 4.0
    gain_cost: float = 2.5


class OperationalCostDynamics:
    """Implements the state transition in the Operational Cost subsystem.

    Supports batched transitions.
    """

    def __init__(self):
        self.params = OperationalCostParams()

    def operational_cost(self, setpoint, velocity, gain):
        """Calculate the current operational cost through equation (6) of the paper."""
        costs = (
            self.params.setpoint_cost * setpoint
            + self.params.velocity_cost * velocity
            + self.params.gain_cost * gain
        )
        return np.exp(costs / 100.0)

    def transition(self, setpoint, velocity, gain, theta_vec):
        """
        Shift the history of operational costs and add the previously current cost.
        """
        # Update history of operational costs for future use in equation (7)
        new_cost = self.operational_cost(setpoint, velocity, gain)
        return np.concatenate([new_cost, theta_vec[..., :-1]], axis=-1)

    @staticmethod
    def convoluted_operational_cost(theta_vec):
        """Calculate the convoluted cost according to equation (7) of the paper."""
        conv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9])
        return theta_vec @ conv
