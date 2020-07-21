import numpy as np
import holoviews as hv
from holoviews import dim, opts

import gym
import gym_industrial

renderer = hv.renderer("bokeh")


def cb_maker():
    env = None
    done = True
    timestep = 0

    def callback():
        nonlocal env, done, timestep
        if env is None:
            env = gym.make("IBFatigue-v0").env

        if done:
            env.reset()
            done = False
            timestep = 0

        action = env.action_space.sample()
        # Drive velocity to 0
        action[0:2] = 0
        _, _, done, info = env.step(action)
        timestep += 1
        info["timestep"] = timestep
        return {k: np.array([v]) for k, v in info.items()}

    return callback


# Define DynamicMap callbacks returning Elements
def env_curve(data):
    curves = {
        k: hv.Curve(data, kdims=["timestep"], vdims=[k])
        for k in "velocity gain fatigue hidden_velocity hidden_gain".split()
    }
    return hv.NdOverlay(curves, kdims=None).opts(legend_position="right", width=600)


get_env_data = cb_maker()
env_stream = hv.streams.Buffer(get_env_data(), length=200)


def callback():
    env_stream.send(get_env_data())


env_dmap = hv.DynamicMap(env_curve, streams=[env_stream])

plot = env_dmap

doc = renderer.server_doc(plot)
doc.add_periodic_callback(callback, 0.2)
