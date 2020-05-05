# pylint:disable=missing-function-docstring,invalid-name,protected-access
# pylint:disable=wrong-import-order,no-value-for-parameter
"""
Industrial Benchmark inspection
"""
import functools

import bokeh
import bokeh.palettes
from bokeh.plotting import figure
import gym
import gym_industrial  # pylint:disable=unused-import
import numpy as np
import streamlit as st


# pylint:disable=pointless-string-statement
"""
# Plot Industrial Benchmark
"""
# pylint:enable=pointless-string-statement


def default_goldstone_figure(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "p" not in kwargs:
            p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
            p.xaxis.axis_label = "phi"
            p.yaxis.axis_label = "effective shift"
            kwargs["p"] = p
        return func(*args, **kwargs)

    return wrapped


@default_goldstone_figure
def goldstone_dynamics_landscape(env, p=None):
    p.title.text = "Mis-Calibration"
    phi = np.linspace(-6, 6, num=1000)
    effective_shift = np.linspace(-1.5, 1.5, num=1000)
    X, Y = np.meshgrid(phi, effective_shift)
    Z = env._goldstone.reward(X, Y)

    palette_size = st.slider("palette size", min_value=3, max_value=256, value=256)
    p.image(
        image=[Z],
        x=-6,
        y=-1.5,
        dw=12,
        dh=3,
        palette=bokeh.palettes.viridis(palette_size),
    )
    return p


@default_goldstone_figure
def goldstone_rmin_and_ropt(env, p=None):
    goldstone = env._goldstone
    phi = np.linspace(-6, 6, num=1000)
    rho_s = goldstone.rho_s(phi)
    r_min = goldstone.r_min(rho_s)
    r_opt = goldstone.r_opt(rho_s)
    p.line(phi, r_min, legend_label="r_min", color="red")
    p.line(phi, r_opt, legend_label="r_opt", color="blue")
    return p


@default_goldstone_figure
def goldstone_rhos(env, p=None):
    goldstone = env._goldstone
    phi = np.linspace(-6, 6, num=1000)
    rho_s = goldstone.rho_s(phi)
    p.line(phi, rho_s, legend_label="rho_s", color="black")
    return p


def main():
    env = gym.make("IBMisCalibration-v0").unwrapped
    p = goldstone_dynamics_landscape(env)
    if st.checkbox("Display rho_s"):
        p = goldstone_rhos(env, p=p)
    if st.checkbox("Display r_min and r_opt"):
        p = goldstone_rmin_and_ropt(env, p=p)
    st.bokeh_chart(p)


if __name__ == "__main__":
    main()
