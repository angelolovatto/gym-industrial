# pylint: disable=missing-module-docstring
import gym

gym.register(
    id="IndustrialBenchmark-v0",
    entry_point="gym_industrial.envs:IndustrialBenchmarkEnv",
    max_episode_steps=1000,
)


gym.register(
    id="IBMisCalibration-v0",
    entry_point="gym_industrial.envs.mis_calibration:MisCalibrationEnv",
    max_episode_steps=1000,
)

gym.register(
    id="IBFatigue-v0",
    entry_point="gym_industrial.envs.fatigue:FatigueEnv",
    max_episode_steps=1000,
)


gym.register(
    id="IBOperationalCost-v0",
    entry_point="gym_industrial.envs.operational_cost:OperationalCostEnv",
    max_episode_steps=1000,
)
