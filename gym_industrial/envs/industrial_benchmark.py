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
import numpy as np
import gym
from gym import spaces

from .ids import IDS


class IndustrialBenchmarkEnv(gym.Env):
    """OpenAI Gym wrapper for Industrial Benchmark.

    Currently only supports a fixed setpoint.
    """

    # pylint: disable=missing-docstring,too-many-instance-attributes

    def __init__(
        self,
        setpoint=50,
        reward_type="classic",
        action_type="continuous",
        observation="visible",
        **ids_kwargs,
    ):
        # pylint:disable=too-many-arguments
        # Setting up the IB environment
        self.setpoint = setpoint
        self._ib = IDS(setpoint, **ids_kwargs)
        # Used to determine whether to return the absolute value or the relative change
        # in the cost function
        self.reward_function = reward_type
        self.action_type = action_type
        self.observation = observation

        # Observation bounds for visible state
        # [setpoint, velocity, gain, shift, fatigue, consumption]
        ob_low = np.array([0, 0, 0, 0, -np.inf, -np.inf])
        ob_high = np.array([100, 100, 100, 100, np.inf, np.inf])
        if self.observation == "markovian":
            # Observation bounds for minimal markovian state
            ob_low = np.concatenate([ob_low, np.array([-np.inf] * 14)])
            ob_high = np.concatenate([ob_high, np.array([np.inf] * 14)])
        elif self.observation == "full":
            # Observation bounds for full state
            ob_low = np.concatenate([ob_low, np.array([-np.inf] * 24)])
            ob_high = np.concatenate([ob_high, np.array([np.inf] * 24)])
        self.observation_space = spaces.Box(
            low=ob_low.astype("f"), high=ob_high.astype("f")
        )

        # Action space and the observation space
        if self.action_type == "discrete":
            # Discrete action space with three different values per steerings for the
            # three steerings ( 3^3 = 27)
            self.action_space = spaces.Discrete(27)

            # A list of all possible actions discretized into [-1,0,1]
            # e.g. [[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0], ... ]
            # Network has 27 outputs and chooses one environmental action out of the
            # discretized 27 possible actions
            self.env_action = []
            for vel in [-1, 0, 1]:
                for gain in [-1, 0, 1]:
                    for shift in [-1, 0, 1]:
                        self.env_action.append([vel, gain, shift])
        elif self.action_type == "continuous":
            # Continuous action space for each steering [-1,1]
            ac_low = np.array([-1, -1, -1])
            self.action_space = spaces.Box(ac_low.astype("f"), -ac_low.astype("f"))
        else:
            raise ValueError(
                "Invalid action type {}. "
                "`action_type` can either be 'discrete' or 'continuous'".format(
                    self.action_type
                )
            )

        self.reward = -self._ib.state["cost"]
        # Alternative reward that returns the improvement or decrease in the cost
        # If the cost function improves/decreases, the reward is positiv
        # If the cost function deteriorates/increases, the reward is negative
        # e.g.: -400 -> -450 = delta_reward of -50
        self.delta_reward = 0

        # smoothed_cost is used as a smoother cost function for monitoring the agent
        # & environment with lower variance
        # Updates itself with 0.95*old_cost + 0.05*new_cost or any other linear
        # combination
        self.smoothed_cost = self._ib.state["cost"]

        self.seed()

    def step(self, action):
        # Executing the action and saving the observation
        if self.action_type == "discrete":
            self._ib.step(self.env_action[action])
        elif self.action_type == "continuous":
            self._ib.step(action)

        # Calculating both the relative reward (improvement or decrease) and updating
        # the reward
        self.delta_reward = self.reward - self._ib.state["cost"]
        self.reward = self._ib.state["cost"]

        # Due to the very high stochasticity a smoothed cost function is easier to
        # follow visually
        self.smoothed_cost = int(
            0.9 * self.smoothed_cost + 0.1 * self._ib.state["cost"]
        )

        # Two reward functions are available:
        # 'classic' which returns the original cost and
        # 'delta' which returns the change in the cost function w.r.t. the previous cost
        if self.reward_function == "classic":
            return_reward = -self._ib.state["cost"]
        elif self.reward_function == "delta":
            return_reward = self.delta_reward
        else:
            raise ValueError(
                "Invalid reward function specification. Use 'classic' for the original "
                "cost function or 'delta' for the change in the cost function between "
                "steps."
            )

        # reward is divided by 100 to improve learning
        return self._get_obs(), return_reward / 100, False, self.minimal_markov_state()

    def _get_obs(self):
        if self.observation == "visible":
            obs = self._ib.visible_state()
        elif self.observation == "markovian":
            obs = self._ib.minimal_markov_state()
        elif self.observation == "full":
            obs = self._ib.full_state()
        return obs.astype(np.float32)

    def reset(self):
        # Resetting the entire environment
        self._ib.reset()
        self.reward = -self._ib.state["cost"]
        return self._get_obs()

    def seed(self, seed=None):
        return self._ib.set_seed(seed)

    def render(self, mode="human"):
        pass

    def minimal_markov_state(self):
        markovian_state_variables = [
            "setpoint",
            "velocity",
            "gain",
            "shift",
            "fatigue",
            "consumption",
            "op_cost_t1",
            "op_cost_t2",
            "op_cost_t3",
            "op_cost_t4",
            "op_cost_t5",
            "op_cost_t6",
            "op_cost_t7",
            "op_cost_t8",
            "op_cost_t9",
            "ml1",
            "ml2",
            "ml3",
            "hv",
            "hg",
        ]
        return dict(zip(markovian_state_variables, self._ib.minimal_markov_state()))
