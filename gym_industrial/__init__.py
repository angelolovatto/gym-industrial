# pylint: disable=missing-module-docstring
from gym.envs.registration import register

register(
    id="IndustrialBenchmark-v0",
    entry_point="gym_industrial.envs:IndustrialBenchmarkEnv",
    max_episode_steps=1000,
)
