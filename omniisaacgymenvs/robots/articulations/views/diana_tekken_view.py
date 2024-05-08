# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch

class DianaTekkenView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "TekkenView",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._palm_centers = RigidPrimView(prim_paths_expr="/World/envs/.*/diana/palm_link_hithand", name="palm_centers_view", reset_xform_properties=False)
        self._index_fingers = RigidPrimView(prim_paths_expr="/World/envs/.*/diana/Right_Index_Phadist", name="right_indices_view", reset_xform_properties=False)
        self._middle_fingers = RigidPrimView(prim_paths_expr="/World/envs/.*/diana/Right_Middle_Phadist", name="right_middles_view", reset_xform_properties=False)
        self._ring_fingers = RigidPrimView(prim_paths_expr="/World/envs/.*/diana/Right_Ring_Phadist", name="right_rings_view", reset_xform_properties=False)
        self._little_fingers = RigidPrimView(prim_paths_expr="/World/envs/.*/diana/Right_Little_Phadist", name="right_littles_view", reset_xform_properties=False)
        self._thumb_fingers = RigidPrimView(prim_paths_expr="/World/envs/.*/diana/Right_Thumb_Phadist", name="right_thumbs_view", reset_xform_properties=False)


    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices
    @property
    def actuated_diana_dof_indices(self):
        return self._actuated_diana_dof_indices
    @property
    def actuated_finger_dof_indices(self):
        return self._actuated_finger_dof_indices
    @property
    def clamped_finger_dof_indices(self):
        return self._clamped_finger_dof_indices

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self.actuated_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",

            # "Right_Index_0", 
            # "Right_Middle_0",
            # "Right_Ring_0",
            # "Right_Little_0",
            # "Right_Thumb_0",

            "Right_Index_1",
            # "Right_Middle_1",
            # "Right_Ring_1",
            # "Right_Little_1",
            # "Right_Thumb_1",

            # "Right_Index_2",
            # "Right_Middle_2",
            # "Right_Ring_2",
            # "Right_Little_2",
            # "Right_Thumb_2"
            ]
        self.actuated_diana_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7"
            ]
        self.actuated_finger_joint_names = [
            "Right_Index_1",
            # "Right_Middle_1",
            # "Right_Ring_1",
            # "Right_Little_1",
            # "Right_Thumb_1",
            ]
        self.clamped_finger_joint_names = [     
            # "Right_Index_1",
            "Right_Middle_1",
            "Right_Ring_1",
            "Right_Little_1",
            "Right_Thumb_1",

            "Right_Index_2",
            "Right_Middle_2",
            "Right_Ring_2",
            "Right_Little_2",
            "Right_Thumb_2"]
        
        self._actuated_dof_indices = list()
        self._actuated_diana_dof_indices = list()
        self._actuated_finger_dof_indices = list()
        self._clamped_finger_dof_indices = list()
        for joint_name in self.actuated_joint_names:
            self._actuated_dof_indices.append(self.get_dof_index(joint_name))
        for joint_name in self.actuated_diana_joint_names:
            self._actuated_diana_dof_indices.append(self.get_dof_index(joint_name))
        for joint_name in self.actuated_finger_joint_names:
            self._actuated_finger_dof_indices.append(self.get_dof_index(joint_name))
        for joint_name in self.clamped_finger_joint_names:
            self._clamped_finger_dof_indices.append(self.get_dof_index(joint_name))

        self._actuated_dof_indices.sort()
        self._actuated_diana_dof_indices.sort()
        self._actuated_finger_dof_indices.sort()
        self._clamped_finger_dof_indices.sort()

    def clamp_joint0_joint1(self, actions):
        actions[:, self.clamped_finger_dof_indices] = actions[:, self.actuated_finger_dof_indices]
        return actions


