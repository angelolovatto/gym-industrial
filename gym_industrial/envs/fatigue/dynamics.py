"""Python implementation of Fatigue state transition."""
from dataclasses import dataclass

import numpy as np


@dataclass
class FatigueParams:
    """Parameters for Fatigue dynamics."""

    action_tolerance: float = 0.05
    amplification: float = 0.05
    amplification_max: float = 5.0
    amplification_start: float = 1.2
    exp_lambda: float = 0.1


class FatigueDynamics:
    """Implements the state transition in the Fatigue subsystem.

    Supports batched transitions.

    Args:
        np_random (np.random.mtrand.RandomState): random number generator for latent
            variables.
    """

    def __init__(self, np_random):
        self.params = FatigueParams()
        self.np_random = np_random

    def transition(self, setpoint, velocity, gain, mu_v, mu_g):
        """Compute one timestep of dynamics transition."""
        # pylint:disable=too-many-arguments
        eff_velocity = self.effective_velocity(velocity, gain, setpoint)
        eff_gain = self.effective_gain(gain, setpoint)

        eta_v, eta_g = self.sample_noise_variables(eff_velocity, eff_gain)

        mu_v = self.update_hidden_velocity(mu_v, eff_velocity, eta_v)
        mu_g = self.update_hidden_gain(mu_g, eff_gain, eta_g)

        alpha = self.sample_alpha(mu_v, eta_v, mu_g, eta_g)

        basic_fatigue = self.basic_fatigue(velocity, gain)
        fatigue = self.fatigue(basic_fatigue, alpha)
        return mu_v, mu_g, fatigue

    def effective_velocity(self, velocity, gain, setpoint):
        """Equation 24."""
        maximum = self.unscaled_eff_velocity(100.0, 0.0, setpoint)
        minimum = self.unscaled_eff_velocity(0.0, 100.0, setpoint)
        return (self.unscaled_eff_velocity(velocity, gain, setpoint) - minimum) / (
            maximum - minimum
        )

    def effective_gain(self, gain, setpoint):
        """Equation 25."""
        maximum = self.unscaled_eff_gain(100, setpoint)
        minimum = self.unscaled_eff_gain(0, setpoint)
        return (self.unscaled_eff_gain(gain, setpoint) - minimum) / (maximum - minimum)

    @staticmethod
    def unscaled_eff_velocity(velocity, gain, setpoint):
        """Equation 26."""
        return (gain + setpoint + 2.0) / (velocity - setpoint + 101.0)

    @staticmethod
    def unscaled_eff_gain(gain, setpoint):
        """Equation 27."""
        return 1.0 / (gain + setpoint + 1)

    def sample_noise_variables(self, eff_velocity, eff_gain):
        """Equations 28, 29."""
        # Noise variables described after equation (27)
        eta_ge = self.np_random.exponential(self.params.exp_lambda)
        eta_ve = self.np_random.exponential(self.params.exp_lambda)
        # Apply the logistic fuction to exponential variables
        eta_ge = 2.0 * (1.0 / (1.0 + np.exp(-eta_ge)) - 0.5)
        eta_ve = 2.0 * (1.0 / (1.0 + np.exp(-eta_ve)) - 0.5)

        eta_gu = self.np_random.rand()
        eta_vu = self.np_random.rand()

        eta_gb = np.float(self.np_random.binomial(1, np.clip(eff_gain, 0.001, 0.999)))
        eta_vb = np.float(
            self.np_random.binomial(1, np.clip(eff_velocity, 0.001, 0.999))
        )

        # Equations (28, 29)
        eta_v = eta_ve + (1 - eta_ve) * eta_vu * eta_vb * eff_velocity
        eta_g = eta_ge + (1 - eta_ge) * eta_gu * eta_gb * eff_gain
        return eta_v, eta_g

    def update_hidden_velocity(self, hidden_velocity, eff_velocity, eta_v):
        """Equation 30."""
        return np.where(
            eff_velocity <= self.params.action_tolerance,
            eff_velocity,
            np.where(
                hidden_velocity >= self.params.amplification_start,
                np.minimum(
                    self.params.amplification_max,
                    self.params.amplification * hidden_velocity,
                ),
                0.9 * hidden_velocity + eta_v / 3.0,
            ),
        )

    def update_hidden_gain(self, hidden_gain, eff_gain, eta_g):
        """Equation 31."""
        return np.where(
            eff_gain <= self.params.action_tolerance,
            eff_gain,
            np.where(
                hidden_gain >= self.params.amplification_start,
                np.minimum(
                    self.params.amplification_max,
                    self.params.amplification * hidden_gain,
                ),
                0.9 * hidden_gain + eta_g / 3.0,
            ),
        )

    def sample_alpha(self, hidden_velocity, eta_v, hidden_gain, eta_g):
        """Equation 23."""
        gauss = self.np_random.normal(2.4, 0.4, size=hidden_velocity.shape)
        return np.where(
            np.maximum(hidden_velocity, hidden_gain) >= self.params.amplification_max,
            1.0 / (1.0 + np.exp(-gauss)),
            np.maximum(eta_v, eta_g),
        )

    @staticmethod
    def basic_fatigue(velocity, gain):
        """Equation 21."""
        return np.maximum(
            0.0, ((30000.0 / ((5 * velocity) + 100)) - 0.01 * (gain ** 2)),
        )

    @staticmethod
    def fatigue(basic_fatigue_, alpha):
        """Equation 22."""
        return (basic_fatigue_ * (1 + 2 * alpha)) / 3.0
