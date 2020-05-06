"""Python implementation of Mis-calibration state transition."""
from dataclasses import dataclass, field

import numpy as np

from .goldstone import PenaltyLandscape


@dataclass
class MisCalibrationParams:
    """Parameters for computing the effective shift in Mis-calibration dynamics."""

    gs_bound: float = 1.5
    gs_setpoint_dependency: float = 0.02
    gs_scale: float = field(init=False)

    def __post_init__(self):
        self.gs_scale = 2.0 * self.gs_bound + 100.0 * self.gs_setpoint_dependency


class MisCalibrationDynamics:
    """Implements the state transition in the Mis-calibration subsystem.

    Supports batched transitions.

    Args:
        safe_zone (float): the radius of the safe zone.
    """

    def __init__(self, safe_zone, number_steps=24):
        self._strongest_penality_abs_idx = self.compute_strongest_penalty_abs_idx(
            number_steps
        )
        self.params = MisCalibrationParams()
        self.goldstone = PenaltyLandscape(safe_zone)

    @property
    def safe_zone(self):
        """Radius of the safe zone."""
        return self.goldstone.safe_zone

    def penalty(self, setpoint, shift, phi):
        """Compute penalty by inferring effective shift and applying Equation (17)."""
        effective_shift = self.effective_shift(setpoint, shift)
        return self.goldstone.penalty(phi, effective_shift)

    def transition(self, setpoint, shift, domain, system_response, phi):
        """Compute one timestep of dynamics transition."""
        # pylint:disable=too-many-arguments
        effective_shift = self.effective_shift(setpoint, shift)
        old_domain = domain
        domain = self.domain(old_domain, effective_shift)
        # (1) if domain change: system_response <- advantageous
        system_response = self.system_response1(system_response, old_domain, domain)
        # (2) compute & apply turn direction
        phi = self.apply_angular_step(domain, phi, system_response, effective_shift)
        # (3) Update system response if necessary
        system_response = self.system_response2(phi, system_response)
        # (4) apply symmetry
        phi = self.apply_symmetry(phi)
        # (5) if phi == 0: reset internal state
        domain, system_response = self.reset_if_needed(
            domain, system_response, phi, effective_shift
        )
        return domain, system_response, phi

    def effective_shift(self, setpoint, shift):
        """Combine setpoint and shift according to Equation (8)."""
        return np.clip(
            self.params.gs_scale * shift / 100.0
            - self.params.gs_setpoint_dependency * setpoint
            - self.params.gs_bound,
            -self.params.gs_bound,
            self.params.gs_bound,
        )

    def domain(self, domain, effective_shift):
        """Apply Equation (9)."""
        return np.where(
            np.abs(effective_shift) <= self.safe_zone, domain, np.sign(effective_shift)
        )

    @staticmethod
    def system_response1(system_response, old_domain, domain):
        """Compute first update in system response according to Equation (10)."""
        return np.where(
            old_domain != domain, np.ones_like(system_response), system_response
        )

    def apply_angular_step(self, domain, phi, system_response, effective_shift):
        """Apply Equation (11).

        Compute the change in phi according to Equation (12). Recall that phi moves in
        discrete unit steps."""
        step = np.where(
            np.abs(effective_shift) <= self.safe_zone,
            # cool down: when effective_shift close to zero
            -np.sign(phi),
            np.where(
                phi == -np.sign(domain) * self._strongest_penality_abs_idx,
                # If phi reaches the left or right limit for positive or negative domain
                # respectively, remain constant
                np.zeros_like(phi),
                # Otherwise, move according to system response and domain
                system_response * np.sign(effective_shift),
            ),
        )
        return phi + step

    def system_response2(self, phi, system_response):
        """Apply Equation (13).

        If the absolute value of direction index phi reaches or exceeds the predefined
        maximum index of 6 (upper right and lower left area in Figure 2), response
        enters state disadvantageous and index phi is turned towards 0.
        """
        return np.where(
            np.abs(phi) >= self._strongest_penality_abs_idx,
            -np.ones_like(system_response),
            system_response,
        )

    def apply_symmetry(self, phi):
        """Apply Equation (14).

        If the absolute value of direction index phi reaches or exceeds the predefined
        maximum index of 6 (upper right and lower left area in Figure 2), response
        enters state disadvantageous and index phi is turned towards 0.
        """
        return np.where(
            np.abs(phi) < self._strongest_penality_abs_idx,
            phi,
            2 * self._strongest_penality_abs_idx
            - (
                (phi + (4 * self._strongest_penality_abs_idx))
                % (4 * self._strongest_penality_abs_idx)
            ),
        )

    def reset_if_needed(self, domain, system_response, phi, effective_shift):
        """Apply Equations (15, 16)."""
        cond = (np.abs(effective_shift) <= self.safe_zone) & (phi == 0)
        domain = np.where(cond, np.ones_like(domain), domain)
        system_response = np.where(cond, np.ones_like(system_response), system_response)
        return domain, system_response

    @staticmethod
    def compute_strongest_penalty_abs_idx(number_steps):
        """Compute the maximum absolute value of phi.

        Using the defaults, this implies that phi is in {-6, ..., 6}.
        """
        if (number_steps < 1) or (number_steps % 4 != 0):
            raise ValueError("number_steps must be positive and integer multiple of 4")

        _strongest_penality_abs_idx = number_steps // 4
        return _strongest_penality_abs_idx
