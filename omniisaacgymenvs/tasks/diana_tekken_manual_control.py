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
from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles, euler_angles_to_quats
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualCuboid, FixedSphere
from omniisaacgymenvs.robots.articulations.diana_tekken import DianaTekken
from omniisaacgymenvs.robots.articulations.views.diana_tekken_view import DianaTekkenView
from omniisaacgymenvs.robots.articulations.utils.kinematic_solver import KinematicsSolver

from abc import abstractmethod, ABC
from omni.isaac.core.tasks import BaseTask

class DianaTekkenManualControlTask(ABC, BaseTask):

    def __init__(
        self,
        name: str
    ) -> None:
        BaseTask.__init__(self, name=name, offset=None)

        self._robot = None
        self._robot_translation = np.array([0.0, 0., 0.], dtype=np.float32)
        self.default_prim_path = "/World/"

        self._num_observations = 60
        self._num_actions = 3

    def get_params(self) -> dict:
            """[summary]

            Returns:
                dict: [description]
            """
            params_representation = dict()
            params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
            return params_representation

    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        scene.add_default_ground_plane()
        self.get_tekken(name="diana_tekken",
                        usd_path="C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/models/diana_tekken/diana_tekken.usd",
                        translation=self._robot_translation)
        self.get_cube()
        self.get_target_sphere()
        self.get_reference_cube()

        super().set_up_scene(scene)

        scene.add(self._robot)
        scene.add(self._sphere)
        scene.add(self._cube)
        scene.add(self._ref_cube)

        self._ik = KinematicsSolver(self._robot)
        self._ik_controller = self._robot.get_articulation_controller()
        robot_base_translation,robot_base_orientation = self._robot.get_world_pose()
        self._ik._kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)
        
    def get_tekken(self, name, usd_path, translation):
        self._robot = DianaTekken(prim_path= self.default_prim_path + name,
                              usd_path=usd_path,
                              name=name,
                              translation=translation)
        
    def get_cube(self):
        self.translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cube_position = np.array([0.5, 0., 0.2], dtype=np.float32)
        self.cube_dimension = np.array([0.6, 1, 0.4], dtype=np.float32)
        self.cube_color = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        self._cube = FixedCuboid(prim_path= self.default_prim_path  + "cube",
                                  name="cube",
                                  translation= self.cube_position,
                                  scale = self.cube_dimension,
                                  color=self.cube_color)
    
    def get_target_sphere(self):
        self._sphere_radius = 0.05
        self._sphere_color = np.array([0.1, 0.9, 0.1], dtype=np.float32)
        self._sphere_position = np.array([0.5, 0., 0.46], dtype=np.float32)
        self._sphere_lower_bound = np.array([0.2, -0.5, 0.46], dtype=np.float32)
        self._sphere_upper_bound = np.array([0.8, 0.5, 0.46], dtype=np.float32)

        self._sphere = DynamicSphere(prim_path= self.default_prim_path + "sphere",
                                  name="sphere",
                                  translation= self._sphere_position,
                                  radius = self._sphere_radius,
                                  color=self._sphere_color,
                                  mass = 0.03)
        
        # sphere.set_collision_enabled(False) # Disable collision as it is used as a target

    def get_reference_cube(self):
        self._ref_cube = VisualCuboid(prim_path= self.default_prim_path  + "ref_cube",
                                  name="ref_cube",
                                  translation= np.array([0.4, 0., 0.8], dtype=np.float32),
                                  scale = np.array([0.02, 0.02, 0.02], dtype=np.float32),
                                  color=np.array([1, 0, 0], dtype=np.float32))
        
    def post_reset(self):
        self.xyz_action , _ = self._ik.compute_end_effector_pose()

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        # implement logic to be performed before physics steps
        pass
    
    def update(self, delta):
        target_pos, target_rot = self._ref_cube.get_world_pose()
        rpy_target = quats_to_euler_angles(target_rot)
        target_pos += delta[:3] * 0.001
        rpy_target[1] += delta[3]*0.001
        target_rot = euler_angles_to_quats(rpy_target)

        self._ref_cube.set_world_pose(position=target_pos, orientation=target_rot)

        actions, succ = self._ik.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_rot)
        if succ:
            self._ik_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")


    def get_observations(self) -> dict:
        # dof_pos = self.diana_tekkens.get_joint_positions(clone=False)
        # dof_vel = self.diana_tekkens.get_joint_velocities(clone=False)
        # hand_pos_world,  self.hand_rot = self.diana_tekkens._palm_centers.get_world_poses(clone=False)
        # index_pos_world, _ = self.diana_tekkens._index_fingers.get_world_poses(clone=False)
        # middle_pos_world, _ = self.diana_tekkens._middle_fingers.get_world_poses(clone=False)
        # ring_pos_world, _ = self.diana_tekkens._ring_fingers.get_world_poses(clone=False)
        # little_pos_world, _ = self.diana_tekkens._little_fingers.get_world_poses(clone=False)
        # thumb_pos_world, _ = self.diana_tekkens._thumb_fingers.get_world_poses(clone=False)

        # target_pos_world, self.target_rot = self._spheres.get_world_poses(clone=False)

        # self.hand_pos = hand_pos_world - self._env_pos
        # self.target_pos = target_pos_world - self._env_pos

        # self.index_pose = index_pos_world - self._env_pos
        # self.middle_pose = middle_pos_world - self._env_pos
        # self.ring_pose = ring_pos_world - self._env_pos
        # self.little_pose = little_pos_world - self._env_pos
        # self.thumb_pose = thumb_pos_world - self._env_pos

        # self.obs_buf[:, :27] = dof_pos
        # self.obs_buf[:, 27:30] = self.hand_pos
        # self.obs_buf[:, 30:33] = self.target_pos
        # self.obs_buf[:, 33:60] = dof_vel
        # # # implement logic to retrieve observation states
        # observations = {self.diana_tekkens.name: {"obs_buf": self.obs_buf}}
        # return observations
        return {
            self._robot.name: {None}
            # self._target.name: {"position": np.array(target_position), "orientation": np.array(target_orientation)},
        }
        pass


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # Reset Diana Tekken robots
        pos = tensor_clamp(
            self.default_dof_pos[self.actuated_dof_indices].unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_actuated_dofs), device=self._device) - 0.5),
            self.diana_tekken_dof_lower_limits[self.actuated_dof_indices],
            self.diana_tekken_dof_upper_limits[self.actuated_dof_indices],
        )

        dof_pos = torch.zeros((num_indices, self.num_diana_tekken_dofs), device=self._device)
        dof_vel = torch.zeros((num_indices, self.num_diana_tekken_dofs), device=self._device)

        dof_pos[:, self.actuated_dof_indices] = pos
        self.diana_tekken_dof_targets[env_ids, :] = dof_pos

        
        self.diana_tekkens.set_joint_positions(dof_pos, indices=indices)
        self.diana_tekkens.set_joint_velocities(dof_vel, indices=indices)
        self.diana_tekkens.set_joint_position_targets(self.diana_tekken_dof_targets[env_ids], indices=indices)

        # Reset target positions
        pos = tensor_clamp(
            self._sphere_position.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), 3), device=self._device) - 0.5),
            self._sphere_lower_bound,
            self._sphere_upper_bound,
        )
        dof_pos = torch.zeros((num_indices, 3), device=self._device)
        dof_pos[:, :] = pos + self._env_pos[env_ids]

        self._spheres.set_world_poses(positions=dof_pos, indices=indices)
        self._spheres.set_velocities(torch.zeros((num_indices, 6)), indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        # Distance to target
        # d = torch.norm(self.hand_pos - self.target_pos, p=2, dim=1)
        # reward = torch.log(1 / (1.0 + d ** 2))



        # # reward = torch.where(torch.norm(self.hand_pos - self.target_pos, p=2, dim=1) < 0.05, reward + 1, reward)
        # reward = torch.where(self.target_pos[:, 2] > 0.5, reward + 10, reward)

        # reward = torch.where(torch.any(self.target_pos[:, :2] >= self._sphere_upper_bound[:2], dim=1), reward - 10, reward)
        # reward = torch.where(torch.any(self.target_pos[:, :2] <= self._sphere_lower_bound[:2], dim=1), reward - 10, reward)

        # self.rew_buf[:] = reward
        pass
        # pass

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # reset if max episode length is exceeded
        # self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(torch.any(self.target_pos[:, :2] >= self._sphere_upper_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(torch.any(self.target_pos[:, :2] <= self._sphere_lower_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)
        pass

