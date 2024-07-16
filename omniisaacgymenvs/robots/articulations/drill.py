from typing import Optional
import numpy as np
import math
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
import omniisaacgymenvs

import carb
from pxr import PhysxSchema
from omni.isaac.core.robots.robot import Robot



class Drill(Robot):

    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Drill",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        
        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        usd_path=f'{omniisaacgymenvs.__path__[0]}/models/drill_trigger.usd'


        self._usd_path = usd_path
        self._name = name
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
        )




