# pylint: disable=missing-docstring,redefined-outer-name,protected-access,unused-import
import gym
import numpy as np
import pytest

import gym_industrial


SETPOINT = (0, 50, 100)
REWARD_TYPE = "classic delta".split()
ACTION_TYPE = "discrete continuous".split()
OBS_TYPE = "visible markovian".split()


@pytest.fixture(
    params=SETPOINT, ids=(f"setpoint({p})" for p in SETPOINT), scope="module"
)
def setpoint(request):
    return request.param


@pytest.fixture(
    params=REWARD_TYPE, ids=(f"reward_type({p})" for p in REWARD_TYPE), scope="module"
)
def reward_type(request):
    return request.param


@pytest.fixture(
    params=ACTION_TYPE, ids=(f"action_type({p})" for p in ACTION_TYPE), scope="module"
)
def action_type(request):
    return request.param


@pytest.fixture(
    params=OBS_TYPE, ids=(f"observation({p})" for p in OBS_TYPE), scope="module"
)
def obs_type(request):
    return request.param


@pytest.fixture(scope="module")
def kwargs(setpoint, reward_type, action_type, obs_type):
    return dict(
        setpoint=setpoint,
        reward_type=reward_type,
        action_type=action_type,
        obs_type=obs_type,
    )


@pytest.fixture(scope="module")
def env(kwargs):
    return gym.make("IndustrialBenchmark-v0", **kwargs)


def test_step(env):
    obs = env.reset()
    assert obs in env.observation_space

    action = env.action_space.sample()
    new_obs, rew, done, info = env.step(action)
    assert new_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    op_cost_keys = [f"op_cost(t-{i})" for i in range(10)]
    assert all(
        k in info and np.isscalar(info[k])
        for k in "setpoint velocity gain shift domain system_response "
        "phi hidden_velocity hidden_gain".split() + op_cost_keys
    )


def test_obs_consistency(env):
    obs, done = env.reset(), False
    assert obs in env.observation_space

    while not done:
        obs, _, done, _ = env.step(env.action_space.sample())
        assert obs in env.observation_space


@pytest.fixture
def classic_reward_ib():
    return lambda c: gym.make(
        "IndustrialBenchmark-v0", **{**c, "reward_type": "classic"}
    )


@pytest.fixture
def delta_reward_ib():
    return lambda c: gym.make("IndustrialBenchmark-v0", **{**c, "reward_type": "delta"})


@pytest.fixture
def non_reward_kwargs(setpoint, action_type, obs_type):
    return dict(setpoint=setpoint, action_type=action_type, obs_type=obs_type,)


def test_reward_type(classic_reward_ib, delta_reward_ib, non_reward_kwargs):
    classic_reward_ib = classic_reward_ib(non_reward_kwargs)
    delta_reward_ib = delta_reward_ib(non_reward_kwargs)

    classic_reward_ib.seed(42)
    classic_reward_ib.reset()
    delta_reward_ib.seed(42)
    delta_reward_ib.reset()

    act = classic_reward_ib.action_space.sample()
    _, rew, _, _ = classic_reward_ib.step(act)
    _, rew2, _, _ = classic_reward_ib.step(act)
    _, _, _, _ = delta_reward_ib.step(act)
    _, rew_, _, _ = delta_reward_ib.step(act)

    assert np.allclose(rew2 - rew, rew_)
