"""Industrial Benchmark as a Gym environment."""
import collections
from dataclasses import dataclass

import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .operational_cost import OperationalCostDynamics
from .mis_calibration import MisCalibrationDynamics
from .fatigue import FatigueDynamics


IndustrialBenchmarkDynamics = collections.namedtuple(
    "IndustrialBenchmarkDynamics", "operational_cost mis_calibration fatigue"
)


@dataclass
class IndustrialBenchmarkParams:
    """Parameters for the Industrial Benchmark."""

    velocity_scale: float = 1
    gain_scale: float = 10
    shift_scale: float = 20 * np.sin(15 * np.pi / 180) / 0.9
    safe_zone: float = np.sin(np.pi * 15 / 180) / 2


class IndustrialBenchmarkEnv(gym.Env):
    """Standalone implementation of the Industrial Benchmark as a Gym environment.

    From the paper:
    > The IB aims at being realistic in the sense that it includes a variety of aspects
    > that we found to be vital in industrial applications like optimization and control
    > of gas and wind turbines. It is not designed to be an approximation of any real
    > system, but to pose the same hardness and complexity. Nevertheless, the process of
    > searching for an optimal action policy on the IB is supposed to resemble the task
    > of finding optimal valve settings for gas turbines or optimal pitch angles and
    > rotor speeds for wind turbines.

    Args:
        setpoint (float): setpoint parameter for the dynamics, as described in the paper
    """

    # pylint:disable=abstract-method

    def __init__(self, setpoint=50):
        super().__init__()
        self.observation_space = Box(
            low=np.array([0] * 4 + [-np.inf] * 2, dtype=np.float32),
            high=np.array([100] * 4 + [np.inf] * 2, dtype=np.float32),
        )
        self.action_space = Box(
            low=np.array([-1] * 3, dtype=np.float32),
            high=np.array([1] * 3, dtype=np.float32),
        )

        self._setpoint = setpoint
        self.params = IndustrialBenchmarkParams()
        self.dynamics = IndustrialBenchmarkDynamics(
            operational_cost=OperationalCostDynamics(),
            mis_calibration=MisCalibrationDynamics(self.params.safe_zone),
            fatigue=None,
        )
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.dynamics.fatigue = FatigueDynamics(self.np_random)
        return [seed]

    def reset(self):
        # pylint:disable=unbalanced-tuple-unpacking
        setpoint = np.array([self._setpoint])
        steerings = self.np_random.uniform(low=0, high=100, size=(3,))
        consumption, fatigue = np.zeros(1), np.zeros(1)
        velocity, gain, _ = np.split(steerings, 3, axis=-1)

        theta_vec = np.full(
            (10,),
            self.dynamics.operational_cost.operational_cost(
                setpoint, velocity, gain
            ).item(),
        )
        # Initial values
        # domain: positive
        # system response: advantageous
        # phi: 0 (center)
        delta_psi_phi = np.array([1, 1, 0])
        mu_v_g = np.zeros(2)

        self.state = np.concatenate(
            [
                setpoint,
                steerings,
                consumption,
                fatigue,
                theta_vec,
                delta_psi_phi,
                mu_v_g,
            ]
        )
        return self._get_obs(self.state)

    def step(self, action):
        assert action in self.action_space

        state = self.state
        self.state = next_state = self._transition_fn(self.state, action)
        reward = self._reward_fn(state, action, next_state).item()
        done = self._terminal(next_state)

        return self._get_obs(next_state), reward, done, {}

    def _transition_fn(self, state, action):
        # pylint:disable=unbalanced-tuple-unpacking
        setpoint, velocity, gain, shift = np.split(state[..., :4], 4, axis=-1)
        theta_vec = state[..., 6:16]
        domain, system_response, phi = np.split(state[..., 16:19], 3, axis=-1)
        mu_v, mu_g = np.split(state[..., 19:], 2, axis=-1)

        velocity, gain, shift = self._apply_action(action, velocity, gain, shift)
        theta_vec = self.dynamics.operational_cost.transition(
            setpoint, velocity, gain, theta_vec
        )
        domain, system_response, phi = self.dynamics.mis_calibration.transition(
            setpoint, shift, domain, system_response, phi
        )
        consumption = self._consumption(setpoint, shift, phi, theta_vec)
        mu_v, mu_g, fatigue = self.dynamics.fatigue.transition(
            setpoint, velocity, gain, mu_v, mu_g
        )

        return np.concatenate(
            [
                setpoint,
                velocity,
                gain,
                shift,
                consumption,
                fatigue,
                theta_vec,
                domain,
                system_response,
                phi,
                mu_v,
                mu_g,
            ],
            axis=-1,
        )

    def _apply_action(self, action, velocity, gain, shift):
        """Apply Equations (2,3,4)."""
        # pylint:disable=unbalanced-tuple-unpacking
        delta_v, delta_g, delta_h = np.split(action, 3, axis=-1)
        velocity = np.clip(velocity + delta_v * self.params.velocity_scale, 0, 100)
        gain = np.clip(gain + delta_g * self.params.gain_scale, 0, 100)
        shift = np.clip(shift + delta_h * self.params.shift_scale, 0, 100)
        return velocity, gain, shift

    def _consumption(self, setpoint, shift, phi, theta_vec):
        """Infer consumption from operational cost and mis-calibration variables."""
        conv_cost = self.dynamics.operational_cost.convoluted_operational_cost(
            theta_vec
        )
        penalty = self.dynamics.mis_calibration.penalty(setpoint, shift, phi)

        c_hat = conv_cost + 25 * penalty
        gauss = self.np_random.normal(0, 1 + 0.02 * c_hat)
        return c_hat + gauss

    @staticmethod
    def _reward_fn(state, action, next_state):
        """Compute Equation (5)."""
        # pylint:disable=unused-argument
        consumption, fatigue = next_state[..., 4], next_state[..., 5]
        return -consumption - 3 * fatigue

    @staticmethod
    def _terminal(_):
        return False

    @staticmethod
    def _get_obs(state):
        return state[..., :6].astype(np.float32)
