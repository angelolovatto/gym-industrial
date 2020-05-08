# pylint: disable=missing-docstring,redefined-outer-name,protected-access,unused-import
import gym
import numpy as np
import pytest

import gym_industrial


@pytest.fixture(scope="module")
def env():
    return gym.make("IBFatigue-v0")


def test_step(env):
    obs = env.reset()
    assert obs in env.observation_space

    action = env.action_space.sample()
    new_obs, rew, done, info = env.step(action)
    assert new_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert all(
        k in info and np.isscalar(info[k])
        for k in "setpoint velocity gain hidden_velocity hidden_gain".split()
    )


def test_obs_consistency(env):
    obs, done = env.reset(), False
    assert obs in env.observation_space

    while not done:
        obs, _, done, _ = env.step(env.action_space.sample())
        assert obs in env.observation_space
