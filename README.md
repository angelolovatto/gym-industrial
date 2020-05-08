# Industrial Benchmark for Gym

`gym-industrial` is a standalone Python re-implementation of the [Industrial Benchmark](https://github.com/siemens/industrialbenchmark) (IB) for OpenAI Gym.
> D. Hein et al., 2017 <br>
> [**A benchmark environment motivated by industrial control problems.**](https://arxiv.org/abs/1709.09480) <br>
> In IEEE Symposium Series on Computational Intelligence (SSCI) (pp. 1-8).

## Installation

```bash
pip install gym-industrial
```

## Environments

To register the environments in Gym, simply import the package at any point before calling `gym.make`.
```python
import gym
import gym_industrial

env = gym.make(<environment id>, **kwargs)
```

The main environment is registered in Gym as `IndustrialBenchmark-v0`. The IB's sub-dynamics have also been implemented as Gym environments. Each contributes with different challenges to the overall task.

| System | Environment ID | Features |
| -------- | -------- | ------ |
| Industrial Benchmark | IndustrialBenchmark-v0 | All of the following
| Operational Cost | IBOperationalCost-v0 | Delayed, blurred, nonlinear rewards |
| Mis-calibration | IBMisCalibration-v0 | Partial observability, non-stationary dynamics |
| Fatigue | IBFatigue-v0 | Heteroscedatisc noise, self-amplifying processes


## Dynamics as Stochastic Computation Graphs
The following are views of the Industrial Benchmark sub-dynamics, plus the reward function, as stochastic computation graphs (SCG).

The graphical notation used and the SCG definition are taken from [Gradient Estimation Using Stochastic Computation Graphs](http://papers.nips.cc/paper/5899-gradient-estimation-using-stochastic-computation-graphs).
> Definition 1 (Stochastic Computation Graph). A directed, acyclic graph, with three types of nodes:
> 1. Input nodes, which are set externally, including the parameters we differentiate with respect to.
> 2. Deterministic nodes, which are functions of their parents.
> 3. Stochastic nodes, which are distributed conditionally on their parents.
Each parent v of a non-input node w is connected to it by a directed edge (v, w).

Squares denote deterministic nodes and circles, stochastic nodes. A special type of deterministic node, denoted by diamonds, indicates that a variable is a cost/reward and thus not part of the observation/state.

Node labels use the notation from the Industrial Benchmark [paper](https://arxiv.org/abs/1709.09480) and correspond to the variables in the equations therein.

### Operational cost
> The sub-dynamics of operational cost are influenced by the external driver setpoint p and two of the three steerings, velocity v and gain g.

The observation of operational cost is delayed and blurred by a convolution of past operational costs. In the graph below, ![\overrightarrow{\theta}](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7B%5Ctheta%7D) denotes a vector of the past 10 values of the hidden operational cost, ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta).

> The motivation for this dynamical behavior is that it is non-linear, it depends on more than one influence, and it is delayed and blurred. All those effects have been observed in industrial applications, like the heating process observable during combustion.

<center><img src=https://i.imgur.com/ZKHaVGD.png width=400></center>

<!-- ![](https://i.imgur.com/ZKHaVGD.png =400x)
 -->

### Mis-calibration dynamics
> The sub-dynamics of mis-calibration are influenced by external driver setpoint p and steering shift h. The goal is to reward an agent to oscillate in h in a pre-defined frequency around a specific operation point determined by setpoint p. Thereby, the reward topology is inspired by an example from quantum physics, namely Goldstone’s ”Mexican hat” potential.

The Goldstone potential-inspired reward is denoted below by the ![m_{t+1}](https://render.githubusercontent.com/render/math?math=m_%7Bt%2B1%7D) node for ease of presentation. Details of the function can be found in the [implementation](gym_industrial/envs/mis_calibration/goldstone.py) or in Appendix B of the paper.

<img src=https://i.imgur.com/VwS8RV9.png height=400>

Below is a visual description, taken from the paper, of the penalty landscape and oscillating dynamics.

<center><img src=https://i.imgur.com/SiDalwC.png width=500></center>

<!-- ![](https://i.imgur.com/SiDalwC.png =500x)
 -->
### Fatigue dynamics
> The sub-dynamics of fatigue are influenced by the same variables as the sub-dynamics of operational cost, i.e., setpoint p, velocity v, and gain g. The IB is designed in such a way that, when changing the steerings velocity v and gain g as to reduce the operational cost, fatigue will be increased, leading to the desired multi-criterial task, with two reward components showing opposite dependencies on the actions.

The following SCG highlights the complex stochasticity of the Fatigue dynamics. The random variables don't have dedicated equations in the paper, but are sampled as follows (![\exp](https://render.githubusercontent.com/render/math?math=%5Cexp) denotes the [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) distribution and ![\sigma](https://render.githubusercontent.com/render/math?math=%5Csigma), the [logistic](https://en.wikipedia.org/wiki/Sigmoid_function) function).

<center><img src=https://i.imgur.com/GT8nngO.png width=300></center>
<!-- $$
\begin{gather*}
\eta^{ve}, \eta^{ge} \sim \sigma(\exp(\lambda)) \\
\eta^{vb}/\eta^{gb} \sim {\rm Binom}(1, v^e)/{\rm Binom}(1, g^e) \\
\eta^{vu}, \eta^{gu} \sim \mathcal{U}(0, 1) \\
\xi \sim \mathcal{N}(2.4, 0.4)
\end{gather*}
$$
 -->

![](https://i.imgur.com/9hoVxh7.png)


### Reward function
> In the real-world tasks that motivated the IB, the reward function has always been known explicitly. In some cases it itself was subject to optimization and had to be adjusted to properly express the optimization goal. For the IB we therefore assume that the reward function is known and all variables influencing it are observable.

<center><img src=https://i.imgur.com/E9Vx5yO.png width=400x></center>
