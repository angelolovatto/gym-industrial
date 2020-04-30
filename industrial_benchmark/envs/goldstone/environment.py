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
from .dynamics import Dynamics


class GoldstoneEnvironment:
    # pylint:disable=missing-docstring

    def __init__(self, number_steps, safe_zone):
        self._dynamics = Dynamics(number_steps, safe_zone)

    @property
    def safe_zone(self):
        return self._dynamics.safe_zone

    def reward(self, phi_idx, effective_shift):
        return self._dynamics.reward(phi_idx, effective_shift)

    def state_transition(self, domain, phi_idx, system_response, effective_shift):
        domain, phi_idx, system_response = self._dynamics.state_transition(
            domain, phi_idx, system_response, effective_shift
        )
        return self.reward(phi_idx, effective_shift), domain, phi_idx, system_response
