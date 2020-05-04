# pylint: disable=missing-module-docstring
import gym

gym.register(
    id="IndustrialBenchmark-v0",
    entry_point="gym_industrial.envs:IndustrialBenchmarkEnv",
    max_episode_steps=1000,
)
