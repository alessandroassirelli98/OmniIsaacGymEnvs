import math

import numpy as np
import torch
import carb
from gym import spaces
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import  euler_angles_to_quats, get_euler_xyz
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, DynamicCuboid, VisualCuboid, FixedSphere
from omniisaacgymenvs.robots.articulations.diana_tekken import DianaTekken
from omniisaacgymenvs.robots.articulations.views.diana_tekken_view import DianaTekkenView
from omniisaacgymenvs.robots.articulations.utils.kinematic_solver import KinematicsSolver
from omniisaacgymenvs.tasks.diana_tekken_task import DianaTekkenTask
from abc import abstractmethod, ABC
from omni.isaac.core.tasks import BaseTask

class DianaTekkenManualControlTask(DianaTekkenTask):

    def __init__(self, name: str, sim_config, env, offset=None) -> None:
        self._num_actions = 4

        DianaTekkenTask.__init__(self, name=name, sim_config=sim_config, env=env, offset=None)
        self.obs_buf = np.zeros(self._num_observations)


    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_reference_cube()
        super().set_up_scene(scene)

        self._ref_cubes = GeometryPrimView(prim_paths_expr="/World/envs/.*/ref_cube", name="ref_cube_view", reset_xform_properties=False)
        scene.add(self._ref_cubes)
        
        scene.add(self._robot)
        
        self._ik = KinematicsSolver(self._robot)
        self._articulation_controller = self._robot.get_articulation_controller()
        robot_base_translation,robot_base_orientation = self._robot.get_world_pose()
        self._ik._kinematics_solver.set_robot_base_pose(np.array(robot_base_translation), np.array(robot_base_orientation))

    def get_reference_cube(self):
        self._ref_cube = VisualCuboid(prim_path= self.default_zero_env_path + "/ref_cube",
                                  name="ref_cube",
                                  translation= torch.tensor([0.4, 0., 0.8], device=self._device),
                                  orientation= euler_angles_to_quats(torch.tensor([0., -np.pi/6, 0.], device=self._device).unsqueeze(0)).squeeze(0),
                                  scale = torch.tensor([0.02, 0.02, 0.02], device=self._device),
                                  color= torch.tensor([1, 0, 0], device=self._device))

    def post_reset(self):
        self.cloned_robot_actions = np.zeros((22))
        super().post_reset()
    
    
    def pre_physics_step(self, actions: np.array) -> None:
        # Move target position and orientation
        target_pos, target_rot = self._ref_cubes.get_world_poses()
        rpy_target = torch.tensor(get_euler_xyz(target_rot)).unsqueeze(0)
        target_pos[0, :3] += actions[:3] * 0.001
        rpy_target[0, 1] += actions[3] * 0.001
        target_rot = euler_angles_to_quats(rpy_target)

        self._ref_cubes.set_world_poses(positions=target_pos, orientations=target_rot)

        robot_actions, succ = self._ik.compute_inverse_kinematics(
            target_position=np.array(target_pos.squeeze(0)),
            target_orientation=np.array(target_rot.squeeze(0)))
        
        if actions[-1] == 1:
            self.cloned_robot_actions[self._robots.actuated_finger_dof_indices] = np.ones(len(self._robots.actuated_finger_dof_indices)) * np.pi/2
        else:
            self.cloned_robot_actions[self._robots.actuated_finger_dof_indices] = np.zeros(len(self._robots.actuated_finger_dof_indices))

        if succ:
            self.cloned_robot_actions[self._robots.actuated_diana_dof_indices] = robot_actions.joint_positions
            robots_actions = torch.tensor((self.cloned_robot_actions.reshape(1,-1)).astype(np.float32))
            robots_actions[:, self.actuated_dof_indices] = (robots_actions[:, self.actuated_dof_indices] - self._robots.get_applied_actions().joint_positions[:, self.actuated_dof_indices])/(self.dt * self.action_scale)
            robots_actions[:, self.actuated_dof_indices] = tensor_clamp(robots_actions[:, self.actuated_dof_indices], -1. * torch.ones(1, self.num_actuated_dofs), 1. * torch.ones(1, self.num_actuated_dofs))
            super().pre_physics_step(robots_actions[:, self.actuated_dof_indices])
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")
        
        



    def get_observations(self) -> dict:
        super().get_observations()


    # def is_done(self) -> None:
    #     pass
