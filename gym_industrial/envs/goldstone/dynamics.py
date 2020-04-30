"""
Copyright 2016 Siemens AG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from enum import Enum

import numpy as np


class Dynamics:
    # pylint:disable=missing-docstring

    alpha = 0.5849
    beta = 0.2924
    kappa = -0.6367

    class Domain(Enum):
        negative = -1
        positive = +1

    class SystemResponse(Enum):
        advantageous = +1
        disadvantageous = -1

    def __init__(self, number_steps, safe_zone):
        self._safe_zone = self._check_safe_zone(safe_zone)
        self._strongest_penality_abs_idx = self.compute_strongest_penalty_abs_idx(
            number_steps
        )

    @staticmethod
    def _check_safe_zone(safe_zone):
        if safe_zone < 0:
            raise ValueError("safe_zone must be non-negative")
        return safe_zone

    @property
    def safe_zone(self):
        return self._safe_zone

    def reset(self):
        return self.Domain.positive, 0, self.SystemResponse.advantageous

    def reward(self, phi, effective_shift):  # pylint: disable=arguments-differ
        rho_s = self._compute_rhos(phi)
        omega = self.omega(rho_s, effective_shift)

        return (
            -self.alpha * omega ** 2
            + self.beta * omega ** 4
            + self.kappa * rho_s * omega
        )

    @staticmethod
    def _compute_rhos(phi):
        """Compute \rho^s as given by Equation (18)."""
        return np.sin(np.pi * phi / 12)

    def omega(self, rho_s, effective_shift):
        """Compute omega as given by Equation (40)."""
        # pylint:disable=invalid-name
        r_opt = self._compute_ropt(rho_s)
        r_min = self._compute_rmin(rho_s)

        mask = np.abs(effective_shift) <= np.abs(r_opt)
        omega = np.empty_like(effective_shift)
        omega[mask] = self._compute_omega1(
            r_min[mask], r_opt[mask], effective_shift[mask]
        )
        omega[~mask] = self._compute_omega2(
            r_min[~mask], r_opt[~mask], effective_shift[~mask]
        )
        return omega

    def _compute_ropt(self, rho_s):
        """Compute r_opt resulting from Equation (43)."""
        varrho = np.sign(rho_s)
        varrho = np.where(varrho == 0, 1.0, varrho)
        return varrho * np.maximum(np.abs(rho_s), 2 * self._safe_zone)

    def _compute_rmin(self, rho_s):
        """Compute r_min resulting from Equation (44)."""
        # pylint:disable=invalid-name
        varrho = np.sign(rho_s)
        q = self._compute_q(rho_s)

        # mask = q < -np.sqrt(1 / 27)
        # r_min = np.empty_like(rho_s)
        # r_min[mask] = self._compute_r_min1(q[mask], varrho[mask])
        # r_min[~mask] = self._compute_r_min2(q[~mask], varrho[~mask])
        r_min = self._compute_r_min2(q, varrho)
        return r_min

    def _compute_q(self, rho_s):
        """Compute q resulting from Equation (46)."""
        return self.kappa * np.abs(rho_s) / (8 * self.beta)

    def _compute_r_min1(self, q, varrho):
        """Compute r_min resulting from the first branch of Equation (44)."""
        # pylint:disable=invalid-name
        u = self._compute_u(q, varrho)
        return (u + 1) / (3 * u)

    @staticmethod
    def _compute_u(q, varrho):
        """Compute u resulting from Equation (45)."""
        # pylint:disable=invalid-name
        base = -varrho * q + np.sqrt(q ** 2 - (1 / 27))
        return np.sign(base) * np.exp(np.log(np.abs(base)) / 3.0)

    @staticmethod
    def _compute_r_min2(q, varrho):
        """Compute r_min resulting from the second branch of Equation (44)."""
        # pylint:disable=invalid-name
        return (
            varrho
            * np.sqrt(4 / 3)
            * np.cos((1 / 3) * np.arccos(-q * (1 / 0.28)))
            # * np.cos((1 / 3) * np.arccos(-q * np.sqrt(27)))
        )

    @staticmethod
    def _compute_omega1(r_min, r_opt, effective_shift):
        """Compute omega resulting from the first branch of Equation (40)."""
        return effective_shift * np.abs(r_min) / np.abs(r_opt)

    @staticmethod
    def _compute_omega2(r_min, r_opt, effective_shift):
        """Compute omega resulting from the second branch of Equation (40)."""
        omega_hat_hat = (2 - np.abs(r_opt)) / (2 - np.abs(r_min))
        ratio_ = (np.abs(effective_shift) - np.abs(r_opt)) / (2 - np.abs(r_opt))
        ratio_to_omega_hat_hat = ratio_ ** omega_hat_hat
        omega_hat = np.abs(r_min) + (2 - np.abs(r_min)) * ratio_to_omega_hat_hat
        omega2 = np.sign(effective_shift) * omega_hat
        return omega2

    def state_transition(self, domain, phi_idx, system_response, effective_shift):

        old_domain = domain

        # (0) compute new domain
        domain = self._compute_domain(old_domain, effective_shift)

        # (1) if domain change: system_response <- advantageous
        # Apply Equation (10)
        if domain != old_domain:
            system_response = self.SystemResponse.advantageous

        # (2) compute & apply turn direction
        # Apply Equation (11)
        phi_idx += self._compute_angular_step(
            domain, phi_idx, system_response, effective_shift
        )

        # (3) Update system response if necessary
        system_response = self._updated_system_response(phi_idx, system_response)

        # (4) apply symmetry
        phi_idx = self._apply_symmetry(phi_idx)

        # (5) if self._Phi_idx == 0: reset internal state
        # Apply Equations (15, 16)
        if (phi_idx == 0) and (abs(effective_shift) <= self._safe_zone):
            domain, phi_idx, system_response = self.reset()

        return domain, phi_idx, system_response

    def _compute_domain(self, domain, effective_shift):
        """Apply Equation (9)"""
        # compute the new domain of control action
        if abs(effective_shift) <= self._safe_zone:
            return domain

        return self.Domain(np.sign(effective_shift))

    def _compute_angular_step(self, domain, phi_idx, system_response, effective_shift):
        """Apply Equation (12).

        Compute the change in phi. Recall that phi moves in discrete unit steps."""
        # cool down: when effective_shift close to zero
        if abs(effective_shift) <= self._safe_zone:  # cool down
            return -np.sign(phi_idx)

        # If phi reaches the left or right limit for positive or negative domain
        # respectively, remain constant
        if phi_idx == -domain.value * self._strongest_penality_abs_idx:
            return 0
        # If phi is in the middle range, move according to system response and domain
        return system_response.value * np.sign(effective_shift)

    def _updated_system_response(self, phi_idx, system_response):
        """Apply Equation (13).

        If the absolute value of direction index phi reaches or exceeds the predefined
        maximum index of 6 (upper right and lower left area in Figure 2), response
        enters state disadvantageous and index phi is turned towards 0.
        """
        if abs(phi_idx) >= self._strongest_penality_abs_idx:
            return self.SystemResponse.disadvantageous

        return system_response

    def _apply_symmetry(self, phi_idx):
        """Apply Equation (14).

        If the absolute value of direction index phi reaches or exceeds the predefined
        maximum index of 6 (upper right and lower left area in Figure 2), response
        enters state disadvantageous and index phi is turned towards 0.
        """
        if abs(phi_idx) < self._strongest_penality_abs_idx:
            return phi_idx

        phi_idx = (phi_idx + (4 * self._strongest_penality_abs_idx)) % (
            4 * self._strongest_penality_abs_idx
        )
        phi_idx = 2 * self._strongest_penality_abs_idx - phi_idx
        return phi_idx

    @staticmethod
    def compute_strongest_penalty_abs_idx(number_steps):
        """Compute the maximum absolute value of phi.

        Using the defaults, this implies that phi is in {-6, ..., 6}.
        """
        if (number_steps < 1) or (number_steps % 4 != 0):
            raise ValueError("number_steps must be positive and integer multiple of 4")

        _strongest_penality_abs_idx = number_steps // 4
        return _strongest_penality_abs_idx
