from typing import Optional
import numpy as np
import math
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

import carb
from pxr import PhysxSchema


class FrankaSimple(Robot):

    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Franka",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "panda_link0/panda_joint1",
            "panda_link1/panda_joint2",
            "panda_link2/panda_joint3",
            "panda_link3/panda_joint4",
            "panda_link4/panda_joint5",
            "panda_link5/panda_joint6",
            "panda_link6/panda_joint7",
        ]

        drive_type = ["angular"] * 7
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]]
        stiffness = [400*np.pi/180] * 7
        damping = [10*np.pi/180] * 7
        max_force = [87, 87, 87, 87, 12, 12, 12]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )
        
        PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])
        
