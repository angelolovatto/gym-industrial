"""
Copyright 2016 Siemens AG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class EffectiveAction:
    # pylint:disable=missing-docstring

    def __init__(self, velocity, gain, setpoint):
        self.setpoint = setpoint
        self.effective_velocity = self.calc_effective_velocity(velocity, gain, setpoint)
        self.effective_gain = self.calc_effective_gain(gain, setpoint)

    def calc_effective_velocity(self, velocity, gain, setpoint):
        min_alpha_unscaled = self.calc_effective_velocity_unscaled(
            self.calceffective_a(100, setpoint), self.calceffective_b(0, setpoint)
        )
        max_alpha_unscaled = self.calc_effective_velocity_unscaled(
            self.calceffective_a(0, setpoint), self.calceffective_b(100, setpoint)
        )
        alpha_unscaled = self.calc_effective_velocity_unscaled(
            self.calceffective_a(velocity, setpoint),
            self.calceffective_b(gain, setpoint),
        )
        return (alpha_unscaled - min_alpha_unscaled) / (
            max_alpha_unscaled - min_alpha_unscaled
        )

    def calc_effective_gain(self, gain, setpoint):
        min_beta_unscaled = self.calc_effective_gain_unscaled(
            self.calceffective_b(100, setpoint)
        )
        max_beta_unscaled = self.calc_effective_gain_unscaled(
            self.calceffective_b(0, setpoint)
        )
        beta_unscaled = self.calc_effective_gain_unscaled(
            self.calceffective_b(gain, setpoint)
        )
        return (beta_unscaled - min_beta_unscaled) / (
            max_beta_unscaled - min_beta_unscaled
        )

    @staticmethod
    def calceffective_a(velocity, setpoint):
        return velocity + 101.0 - setpoint

    @staticmethod
    def calceffective_b(gain, setpoint):
        return gain + 1.0 + setpoint

    @staticmethod
    def calc_effective_velocity_unscaled(effective_a, effective_b):
        return (effective_b + 1.0) / effective_a

    @staticmethod
    def calc_effective_gain_unscaled(effective_b):
        return 1.0 / effective_b

    def get_effective_velocity(self):
        return self.effective_velocity

    def get_effective_gain(self):
        return self.effective_gain
