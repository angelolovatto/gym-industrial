# pylint: disable=missing-docstring,redefined-outer-name,protected-access,unused-import
import gym
import numpy as np
import pytest

import gym_industrial


@pytest.fixture
def env_cls():
    return lambda c: gym.make("IndustrialBenchmark-v0", **c)


@pytest.fixture
def classic_reward_ib(env_cls):
    return env_cls({"reward_type": "classic"})


@pytest.fixture
def delta_reward_ib(env_cls):
    return env_cls({"reward_type": "delta"})


def test_reward_type(classic_reward_ib, delta_reward_ib):
    classic_reward_ib.seed(42)
    classic_reward_ib.reset()
    delta_reward_ib.seed(42)
    delta_reward_ib.reset()

    _, rew, _, _ = classic_reward_ib.step([0.5] * 3)
    _, rew2, _, _ = classic_reward_ib.step([0.5] * 3)
    _, _, _, _ = delta_reward_ib.step([0.5] * 3)
    _, rew_, _, _ = delta_reward_ib.step([0.5] * 3)

    assert np.allclose(rew2 - rew, rew_)
