# pylint: disable=missing-docstring,redefined-outer-name,protected-access,unused-import
import gym
import numpy as np
import pytest

import gym_industrial


SETPOINT = (0, 50, 100)
REWARD_TYPE = "classic delta".split()
ACTION_TYPE = "discrete continuous".split()
OBSERVATION = "visible markovian full".split()


@pytest.fixture(params=SETPOINT, ids=(f"setpoint({p})" for p in SETPOINT))
def setpoint(request):
    return request.param


@pytest.fixture(params=REWARD_TYPE, ids=(f"reward_type({p})" for p in REWARD_TYPE))
def reward_type(request):
    return request.param


@pytest.fixture(params=ACTION_TYPE, ids=(f"action_type({p})" for p in ACTION_TYPE))
def action_type(request):
    return request.param


@pytest.fixture(params=OBSERVATION, ids=(f"observation({p})" for p in OBSERVATION))
def observation(request):
    return request.param


@pytest.fixture
def kwargs(setpoint, reward_type, action_type, observation):
    return dict(
        setpoint=setpoint,
        reward_type=reward_type,
        action_type=action_type,
        observation=observation,
    )


@pytest.fixture
def env(kwargs):
    return gym.make("IndustrialBenchmark-v0", **kwargs)


def test_env_interaction_loop(env):
    obs = env.reset()
    assert obs in env.observation_space

    action = env.action_space.sample()
    new_obs, rew, done, info = env.step(action)
    assert new_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
