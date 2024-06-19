from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class FrankaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FrankaView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_link7", name="hands_view", reset_xform_properties=False
        )
        self._palm_centers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/tekken/palm_link_hithand", name="palm_centers_view", reset_xform_properties=False)

        # self._lfingers = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/franka/panda_leftfinger", name="lfingers_view", reset_xform_properties=False
        # )
        # self._rfingers = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/franka/panda_rightfinger",
        #     name="rfingers_view",
        #     reset_xform_properties=False,
        # )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        self.actuated_joint_names = [
                                    "panda_joint1",
                                    "panda_joint2",
                                    "panda_joint3",
                                    "panda_joint4",
                                    "panda_joint5",
                                    "panda_joint6",
                                    "panda_joint7",

                                    "Right_Index_1",
                                    "Right_Middle_1",
                                    "Right_Ring_1",
                                    "Right_Little_1",
                                    "Right_Thumb_1",

            ]
        
        self.clamp_drive_joint_names = [
            "Right_Index_1",
            "Right_Middle_1",
            "Right_Ring_1",
            "Right_Little_1",
            "Right_Thumb_1",
        ]

        self.clamped_joint_names = [     
            "Right_Index_2",
            "Right_Middle_2",
            "Right_Ring_2",
            "Right_Little_2",
            "Right_Thumb_2",

            "Right_Index_3",
            "Right_Middle_3",
            "Right_Ring_3",
            "Right_Little_3",
            "Right_Thumb_3"]
        
        self._actuated_dof_indices = list()
        self._clamped_dof_indices = list()
        self._clamp_drive_dof_indices = list()
        for joint_name in self.actuated_joint_names:
            self._actuated_dof_indices.append(self.get_dof_index(joint_name))
        for joint_name in self.clamped_joint_names:
            self._clamped_dof_indices.append(self.get_dof_index(joint_name))
        for joint_name in self.clamp_drive_joint_names:
            self._clamp_drive_dof_indices.append(self.get_dof_index(joint_name))

        # self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), self.get_dof_index("panda_finger_joint2")]

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices
    
    @property
    def clamped_dof_indices(self):
        return self._clamped_dof_indices
    
    @property
    def clamp_drive_dof_indices(self):
        return self._clamp_drive_dof_indices
