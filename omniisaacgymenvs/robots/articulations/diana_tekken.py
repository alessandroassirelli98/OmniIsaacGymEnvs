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


class DianaTekken(Robot):

    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Tekken",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        if usd_path is None:
            print("Need to reference a usd Tekken file !")

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
            "base/joint_1",
            "link_1/joint_2",
            "link_2/joint_3",
            "link_3/joint_4",
            "link_4/joint_5",
            "link_5/joint_6",
            "link_6/joint_7",

            # "base_link_hithand/Right_Index_0", 
            # "base_link_hithand/Right_Middle_0",
            # "base_link_hithand/Right_Ring_0",
            # "base_link_hithand/Right_Little_0",
            # "base_link_hithand/Right_Thumb_0",

            # "Right_Index_Basecover/Right_Index_1",
            # "Right_Middle_Basecover/Right_Middle_1",
            # "Right_Ring_Basecover/Right_Ring_1",
            # "Right_Little_Basecover/Right_Little_1",
            # "Right_Thumb_Basecover/Right_Thumb_1",

            # "Right_Index_Phaprox/Right_Index_2",
            # "Right_Middle_Phaprox/Right_Middle_2",
            # "Right_Ring_Phaprox/Right_Ring_2",
            # "Right_Little_Phaprox/Right_Little_2",
            # "Right_Thumb_Phaprox/Right_Thumb_2",

            # These joints are coupled
            # "Right_Index_Phamed/Right_Index_3",
            # "Right_Middle_Phamed/Right_Middle_3",
            # "Right_Ring_Phamed/Right_Ring_3",
            # "Right_Little_Phamed/Right_Little_3",
            # "Right_Thumb_Phamed/Right_Thumb_3",

        ]

        drive_type = ["angular"] * 7
        # default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0. for _ in range(20)]
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]]
        # stiffness = [400*np.pi/180] * 7 + [0.05, 0.05, 0.05, 0.05] * 5
        stiffness = [400*np.pi/180] * 7
        # damping = [80*np.pi/180] * 7 + [0.0009375, 0.000625, 0.000625, 0.000625] * 5
        damping = [80*np.pi/180] * 7
        # max_force = [87, 87, 87, 87, 12, 12, 12] + [10, 1.5, 0.6, 0.3] * 5
        max_force = [87, 87, 87, 87, 12, 12, 12]
        # max_velocity =  [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] +  [3.14 for _ in range(20)]
        max_velocity =  [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]]

        # STICK WITH THE URDF DRIVE PARAMETERS

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
        
