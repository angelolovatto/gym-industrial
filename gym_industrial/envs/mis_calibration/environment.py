"""Standalone mis-calibration subsystem as a Gym environment."""
import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .dynamics import MisCalibrationDynamics


class MisCalibrationEnv(gym.Env):
    """Standalone mis-calibration subsystem as a Gym environment.

    From the paper:
    > The sub-dynamics of mis-calibration are influenced by external driver setpoint p
    > and steering shift h. The goal is to reward an agent to oscillate in h in a pre-
    > -defined frequency around a specific operation point determined by setpoint p.
    > Thereby, the reward topology is inspired by an example from quantum physics,
    > namely Goldstone’s ”Mexican hat” potential.

    Args:
        setpoint (float): setpoint parameter for the dynamics, as described in the paper
        safe_zone (float): the radius of the safe zone.
    """

    # pylint:disable=abstract-method
    action_scale = 20 * np.sin(15 * np.pi / 180) / 0.9

    def __init__(self, setpoint=50, safe_zone=None):
        super().__init__()
        self.observation_space = Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([100, 100], dtype=np.float32),
        )
        self.action_space = Box(
            low=np.array([-1], dtype=np.float32), high=np.array([1], dtype=np.float32)
        )

        self._setpoint = setpoint
        safe_zone = safe_zone or np.sin(np.pi * 15 / 180) / 2
        self._dynamics = MisCalibrationDynamics(safe_zone)
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        setpoint = np.array([self._setpoint])
        shift = self.np_random.uniform(low=0, high=100, size=(1,))
        # Initial values
        # domain: positive
        # system response: advantageous
        # phi: 0 (center)
        hidden_state = np.array([1, 1, 0])
        self.state = np.concatenate([setpoint, shift, hidden_state])
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
        setpoint, shift, domain, system_response, phi = np.split(state, 5, axis=-1)

        shift = self._apply_action(action, shift)
        domain, system_response, phi = self._dynamics.transition(
            setpoint, shift, domain, system_response, phi
        )
        return np.concatenate([setpoint, shift, domain, system_response, phi], axis=-1)

    def _apply_action(self, action, shift):
        """Apply Equation (4)."""
        return np.clip(shift + action * self.action_scale, 0, 100)

    def _reward_fn(self, state, action, next_state):
        # pylint:disable=unused-argument
        setpoint, shift = next_state[..., 0], next_state[..., 1]
        phi = next_state[..., -1]
        return -self._dynamics.penalty(setpoint, shift, phi)

    @staticmethod
    def _terminal(_):
        return False

    @staticmethod
    def _get_obs(state):
        return state[..., :2].astype(np.float32)
