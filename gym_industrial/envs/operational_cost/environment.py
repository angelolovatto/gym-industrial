"""Standalone operational cost subsystem as a Gym environment."""
import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .dynamics import OperationalCostDynamics


class OperationalCostEnv(gym.Env):
    """Standalone operational cost subsystem as a Gym environment.

    From the paper:
    > The sub-dynamics of operational cost are influenced by
    > the external driver setpoint p and two of the three steerings, velocity v and
    > gain g.

    The observation of operational cost is delayed and blurred by a convolution of past
    operational costs.

    > The motivation for this dynamical behavior is that it is non- linear, it depends
    > on more than one influence, and it is delayed and blurred. All those effects have
    > been observed in industrial applications, like the heating process observable
    > during combustion.

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
        self._dynamics = OperationalCostDynamics()
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        setpoint = np.array([self._setpoint])
        velocity = self.np_random.uniform(low=0, high=100, size=(1,))
        gain = self.np_random.uniform(low=0, high=100, size=(1,))
        theta_vec = np.full(
            (10,), self._dynamics.operational_cost(setpoint, velocity, gain).item()
        )
        self.state = np.concatenate([setpoint, velocity, gain, theta_vec])
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
        visible, theta_vec = state[..., :3], state[..., 3:]
        setpoint, velocity, gain = np.split(visible, 3, axis=-1)

        velocity, gain = self._apply_action(action, velocity, gain)
        theta_vec = self._dynamics.transition(setpoint, velocity, gain, theta_vec)
        return np.concatenate([setpoint, velocity, gain, theta_vec], axis=-1)

    @staticmethod
    def _apply_action(action, velocity, gain):
        """Apply Equations (2,3)."""
        # pylint:disable=unbalanced-tuple-unpacking
        delta_v, delta_g = np.split(action, 2, axis=-1)
        velocity = np.clip(velocity + delta_v, 0, 100)
        gain = np.clip(gain + delta_g * 10, 0, 100)
        return velocity, gain

    def _reward_fn(self, state, action, next_state):
        # pylint:disable=unused-argument
        theta_vec = next_state[..., 3:]
        return self._dynamics.convoluted_operational_cost(theta_vec)

    @staticmethod
    def _terminal(_):
        return False

    @staticmethod
    def _get_obs(state):
        return state[..., :3].astype(np.float32)
