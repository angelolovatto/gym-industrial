"""Goldstone Potential Based Equations as described in Appendix B."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GoldstoneParams:
    """Parameters for Goldstone Potential."""

    alpha: float = 0.5849
    beta: float = 0.2924
    kappa: float = -0.6367


class PenaltyLandscape:
    """
    Implements the penalty of the mis-calibration reward component by a so-called
    linearly biased Goldstone potential.

    Args:
       safe_zone (float): the radius of the safe zone.
    """

    # pylint:disable=invalid-name

    def __init__(self, safe_zone):
        self._safe_zone = self._check_safe_zone(safe_zone)
        self.params = GoldstoneParams()

    @staticmethod
    def _check_safe_zone(safe_zone):
        if safe_zone < 0:
            raise ValueError("safe_zone must be non-negative")
        return safe_zone

    @property
    def safe_zone(self):
        """The radius of the safe zone"""
        return self._safe_zone

    def penalty(self, phi, effective_shift):
        """Compute m(\\phi, h^e) as given by Equation (17)."""
        rho_s = self.rho_s(phi)
        omega = self.omega(rho_s, effective_shift)

        return (
            -self.params.alpha * omega ** 2
            + self.params.beta * omega ** 4
            + self.params.kappa * rho_s * omega
        )

    @staticmethod
    def rho_s(phi):
        """Compute \\rho^s as given by Equation (18)."""
        return np.sin(np.pi * phi / 12)

    def omega(self, rho_s, effective_shift):
        """Compute omega as given by Equation (40)."""
        r_opt = self.r_opt(rho_s)
        r_min = self.r_min(rho_s)

        mask = np.abs(effective_shift) <= np.abs(r_opt)
        omega = np.empty_like(effective_shift)
        omega[mask] = self.omega1(r_min[mask], r_opt[mask], effective_shift[mask])
        omega[~mask] = self.omega2(r_min[~mask], r_opt[~mask], effective_shift[~mask])
        return omega

    def r_opt(self, rho_s):
        """Compute r_opt resulting from Equation (43)."""
        varrho = np.sign(rho_s)
        varrho = np.where(varrho == 0, 1.0, varrho)
        return varrho * np.maximum(np.abs(rho_s), 2 * self._safe_zone)

    def r_min(self, rho_s):
        """Compute r_min resulting from Equation (44)."""
        varrho = np.sign(rho_s)
        q = self.equation_46(rho_s)

        mask = q < -np.sqrt(1 / 27)
        r_min = np.empty_like(rho_s)
        r_min[mask] = self.r_min1(q[mask], varrho[mask])
        r_min[~mask] = self.r_min2(q[~mask], varrho[~mask])
        return r_min

    def equation_46(self, rho_s):
        """Compute q resulting from Equation (46)."""
        return self.params.kappa * np.abs(rho_s) / (8 * self.params.beta)

    def r_min1(self, q, varrho):
        """Compute r_min resulting from the first branch of Equation (44)."""
        u = self.equation_45(q, varrho)
        return u + 1 / (3 * u)

    @staticmethod
    def equation_45(q, varrho):
        """Compute u resulting from Equation (45)."""
        base = -varrho * q + np.sqrt(q ** 2 - (1 / 27))
        return np.sign(base) * np.exp(np.log(np.abs(base)) / 3.0)

    @staticmethod
    def r_min2(q, varrho):
        """Compute r_min resulting from the second branch of Equation (44)."""
        return varrho * np.sqrt(4 / 3) * np.cos((1 / 3) * np.arccos(-q * np.sqrt(27)))

    @staticmethod
    def omega1(r_min, r_opt, effective_shift):
        """Compute omega resulting from the first branch of Equation (40)."""
        return effective_shift * np.abs(r_min) / np.abs(r_opt)

    @staticmethod
    def omega2(r_min, r_opt, effective_shift):
        """Compute omega resulting from the second branch of Equation (40)."""
        omega_hat_hat = (2 - np.abs(r_opt)) / (2 - np.abs(r_min))
        ratio_ = (np.abs(effective_shift) - np.abs(r_opt)) / (2 - np.abs(r_opt))
        ratio_to_omega_hat_hat = ratio_ ** omega_hat_hat
        omega_hat = np.abs(r_min) + (2 - np.abs(r_min)) * ratio_to_omega_hat_hat
        omega2 = np.sign(effective_shift) * omega_hat
        return omega2
