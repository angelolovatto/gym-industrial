"""Industrial Benchmark as a Gym environment."""
import collections
from dataclasses import dataclass

import gym
from gym.spaces import Box, Discrete
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

    reward_type: str = None
    action_type: str = None
    obs_type: str = None


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

    Currently only supports a fixed setpoint.

    Args:
        setpoint (float): setpoint parameter for the dynamics, as described in the paper
        reward_type (str): type of the reward function. Either 'classic' or 'delta'
        obs_type (str): type of the observation. Either 'visible' or 'markovian'
    """

    # pylint:disable=abstract-method

    def __init__(
        self,
        setpoint=50,
        reward_type="classic",
        action_type="continuous",
        obs_type="visible",
    ):
        super().__init__()
        self._setpoint = setpoint
        self.params = IndustrialBenchmarkParams(
            reward_type=reward_type, action_type=action_type, obs_type=obs_type
        )

        if self.params.obs_type == "visible":
            self.observation_space = Box(
                low=np.array([0] * 4 + [-np.inf] * 2, dtype=np.float32),
                high=np.array([100] * 4 + [np.inf] * 2, dtype=np.float32),
            )
        elif self.params.obs_type == "markovian":
            self.observation_space = Box(
                low=np.array(
                    [0] * 4 + [-np.inf] * 11 + [-1, -1, -6] + [-np.inf] * 2,
                    dtype=np.float32,
                ),
                high=np.array(
                    [100] * 4 + [np.inf] * 11 + [1, 1, 6] + [np.inf] * 2,
                    dtype=np.float32,
                ),
            )
        else:
            raise ValueError(
                f"Invalid observation type {self.params.obs_type}. "
                "`obs_type` can either be 'visible' or 'markovian'"
            )

        if self.params.action_type == "continuous":
            self.action_space = Box(
                low=np.array([-1] * 3, dtype=np.float32),
                high=np.array([1] * 3, dtype=np.float32),
            )
        elif self.params.action_type == "discrete":
            # Discrete action space with three different values per steerings for the
            # three steerings (3^3 = 27)
            self.action_space = Discrete(27)
        else:
            raise ValueError(
                f"Invalid action type {self.params.action_type}. "
                "`action_type` can either be 'discrete' or 'continuous'"
            )

        self.dynamics = None
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.dynamics = IndustrialBenchmarkDynamics(
            operational_cost=OperationalCostDynamics(),
            mis_calibration=MisCalibrationDynamics(self.params.safe_zone),
            fatigue=FatigueDynamics(self.np_random),
        )
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
        action = self._get_action(action)

        state = self.state
        self.state = next_state = self._transition_fn(self.state, action)
        reward = self._reward_fn(state, action, next_state).item()
        done = self._terminal(next_state)

        return self._get_obs(next_state), reward, done, self._get_info()

    def _get_obs(self, state):
        visible = state[..., :6]
        if self.params.obs_type == "visible":
            return visible.astype(np.float32)

        # current operational cost can be inferred from setpoint, velocity, and gain
        oc_latent = state[..., 6:16]
        oc_latent = oc_latent[..., 1:]

        mc_latent = state[..., 16:19]
        fat_latent = state[..., 19:]
        return np.concatenate(
            [visible, oc_latent, mc_latent, fat_latent], axis=-1
        ).astype(np.float32)

    def _get_action(self, action):
        if self.params.action_type == "continuous":
            return action
        ternary = np.array(
            [(action // 9) % 3, (action // 3) % 3, action % 3], dtype=np.float32
        )
        return ternary - 1

    def _get_info(self):
        # pylint:disable=unbalanced-tuple-unpacking
        state = self.state
        setpoint, velocity, gain, shift = np.split(state[..., :4], 4, axis=-1)
        consumption, fatigue = state[..., 4], state[..., 5]
        theta_vec = state[..., 6:16]
        domain, system_response, phi = np.split(state[..., 16:19], 3, axis=-1)
        mu_v, mu_g = state[..., 19], state[..., 20]
        return {
            "setpoint": setpoint,
            "velocity": velocity,
            "gain": gain,
            "shift": shift,
            "consumption": consumption,
            "fatigue": fatigue,
            "op_cost_history": theta_vec,
            "domain": domain,
            "system_response": system_response,
            "phi": phi,
            "hidden_velocity": mu_v,
            "hidden_gain": mu_g,
        }

    def _transition_fn(self, state, action):
        # pylint:disable=unbalanced-tuple-unpacking
        setpoint, velocity, gain, shift = np.split(state[..., :4], 4, axis=-1)
        theta_vec = state[..., 6:16]
        domain, system_response, phi = np.split(state[..., 16:19], 3, axis=-1)
        mu_v, mu_g = state[..., 19], state[..., 20]

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

    def _reward_fn(self, state, action, next_state):
        """Compute Equation (5)."""
        # pylint:disable=unused-argument
        consumption, fatigue = next_state[..., 4], next_state[..., 5]
        reward = -consumption - 3 * fatigue

        if self.params.reward_type == "delta":
            old_consumption, old_fatigue = state[..., 4], state[..., 5]
            old_reward = -old_consumption - 3 * old_fatigue
            reward -= old_reward

        # reward is divided by 100 to improve learning
        return reward / 100

    @staticmethod
    def _terminal(_):
        return False
