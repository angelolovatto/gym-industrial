"""Standalone fatigue subsystem as a Gym environment."""
import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .dynamics import FatigueDynamics


class FatigueEnv(gym.Env):
    """Standalone fatigue subsystem as a Gym environment.

    From the paper:
    > The sub-dynamics of fatigue are influenced by the same variables as the sub-
    > -dynamics of operational cost, i.e., setpoint p, velocity v, and gain g. The IB is
    > designed in such a way that, when changing the steerings velocity v and gain g as
    > to reduce the operational cost, fatigue will be increased, leading to the desired
    > multi-criterial task, with two reward- components showing opposite dependencies on
    > the actions.

    Args:
        setpoint (float): setpoint parameter for the dynamics, as described in the paper
    """

    # pylint:disable=abstract-method

    def __init__(self, setpoint=50):
        super().__init__()
        self.observation_space = Box(
            low=np.array([0] * 3, dtype=np.float32),
            high=np.array([100] * 3, dtype=np.float32),
        )
        self.action_space = Box(
            low=np.array([-1] * 2, dtype=np.float32),
            high=np.array([1] * 2, dtype=np.float32),
        )

        self._setpoint = setpoint
        self._dynamics = None
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._dynamics = FatigueDynamics(self.np_random)
        return [seed]

    def reset(self):
        setpoint = np.array([self._setpoint])
        velocity = self.np_random.uniform(low=0, high=100, size=(1,))
        gain = self.np_random.uniform(low=0, high=100, size=(1,))
        mu_v = np.zeros(1)
        mu_g = np.zeros(1)
        fatigue = np.zeros(1)
        self.state = np.concatenate([setpoint, velocity, gain, mu_v, mu_g, fatigue])
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
        setpoint, velocity, gain, mu_v, mu_g, _ = np.split(state, 6, axis=-1)

        velocity, gain = self._apply_action(action, velocity, gain)
        mu_v, mu_g, fatigue = self._dynamics.transition(
            setpoint, velocity, gain, mu_v, mu_g
        )
        return np.concatenate([setpoint, velocity, gain, mu_v, mu_g, fatigue], axis=-1)

    @staticmethod
    def _apply_action(action, velocity, gain):
        """Apply Equations (2,3)."""
        # pylint:disable=unbalanced-tuple-unpacking
        delta_v, delta_g = np.split(action, 2, axis=-1)
        velocity = np.clip(velocity + delta_v, 0, 100)
        gain = np.clip(gain + delta_g * 10, 0, 100)
        return velocity, gain

    @staticmethod
    def _reward_fn(state, action, next_state):
        # pylint:disable=unused-argument
        return next_state[..., -1]

    @staticmethod
    def _terminal(_):
        return False

    @staticmethod
    def _get_obs(state):
        return state[..., :3].astype(np.float32)
