# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.motion_generation import ArticulationKinematicsSolver, interface_config_loader, LulaKinematicsSolver
from omni.isaac.core.articulations import Articulation
from typing import Optional


class KinematicsSolver(ArticulationKinematicsSolver):
    """Kinematics Solver for Franka robot.  This class loads a LulaKinematicsSovler object

    Args:
        robot_articulation (Articulation): An initialized Articulation object representing this Franka
        end_effector_frame_name (Optional[str]): The name of the Franka end effector.  If None, an end effector link will
            be automatically selected.  Defaults to None.
    """

    def __init__(self, articulation) -> None:
        # Load Diana URDF
        urdf_dir = "C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/models/diana.urdf"
        descriptor_dir = "C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/models/robot_descriptor.yaml"

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = descriptor_dir,
            urdf_path = urdf_dir
        )

        ee_name = "link_7"
        ArticulationKinematicsSolver.__init__(self, articulation, self._kinematics_solver, ee_name)
        return
