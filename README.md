# Industrial Benchmark for Gym

`gym-industrial` is a standalone Python re-implementation of the [Industrial Benchmark](https://github.com/siemens/industrialbenchmark) (IB) for OpenAI Gym.

## Installation

```bash
pip install gym-industrial
```

## Environments

The IB's subdynamics have also been implemented as Gym environments.

| System | environment id |
| -------- | -------- |
| Industrial Benchmark | IndustrialBenchmark-v0 |
| Operational Cost | IBOperationalCost-v0 |
| Mis-calibration | IBMisCalibration-v0 |
| Fatigue | IBFatigue-v0 |


## Subdynamics Stochastic Computation Graphs
The following are views of the Industrial Benchmark subdynamics, plus the reward function, as stochastic computation graphs (SCG).

The graph notation used and the SCG definition are taken from [Gradient Estimation Using Stochastic Computation Graphs](http://papers.nips.cc/paper/5899-gradient-estimation-using-stochastic-computation-graphs). Squares denote deterministic nodes and circles, stochastic nodes.
> Definition 1 (Stochastic Computation Graph). A directed, acyclic graph, with three types of nodes:
> 1. Input nodes, which are set externally, including the parameters we differentiate with respect to.
> 2. Deterministic nodes, which are functions of their parents.
> 3. Stochastic nodes, which are distributed conditionally on their parents.
Each parent v ofa non-input node w is connected to it by a directed edge (v, w).

Node labels use the notation from the Industrial Benchmark [paper](https://arxiv.org/abs/1709.09480).

### Mis-calibration dynamics
![](https://i.imgur.com/XSwNy6f.png)

### Fatigue dynamics
![](https://i.imgur.com/SnE3KKs.png)

### Operational cost
![](https://i.imgur.com/TimKXjK.png)

### Reward function
![](https://i.imgur.com/E9Vx5yO.png)
