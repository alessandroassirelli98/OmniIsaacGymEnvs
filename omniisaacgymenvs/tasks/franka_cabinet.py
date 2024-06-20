# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
import omniisaacgymenvs
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere, DynamicCuboid
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.drill import Drill
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView, XFormPrimView
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom


class FrankaCabinetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 32
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.reward_terms_log = {}
        self.reward_weights_log = {}
        self.robots_to_log = []


        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.ik_velocity = self._task_cfg["env"]["ikVelocity"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.reward_weights_log["distReward"] =  self._task_cfg["env"]["distRewardScale"]

        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.reward_weights_log["rotReward"] = self._task_cfg["env"]["rotRewardScale"]
        
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.reward_weights_log["aroundHandleReward"] = self._task_cfg["env"]["aroundHandleRewardScale"]

        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.reward_weights_log["openReward"] = self._task_cfg["env"]["openRewardScale"]

        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.reward_weights_log["fingerDistReward"] = self._task_cfg["env"]["fingerDistRewardScale"]

        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.reward_weights_log["actionPenalty"] = self._task_cfg["env"]["actionPenaltyScale"]


        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]
        self.reward_weights_log["fingerCloseReward"] = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.finger_open_reward_scale = self._task_cfg["env"]["fingerOpenRewardScale"]
        self.reward_weights_log["fingerOpenReward"] = self._task_cfg["env"]["fingerOpenRewardScale"]

        self.fail_penalty = self._task_cfg["env"]["failPenalty"]



        self.goal_achieved_bonus = self._task_cfg["env"]["goalAchievedBonus"]

    def set_up_scene(self, scene) -> None:
        self.get_franka()
        # self.get_cabinet()
        self.get_drill()
        self.get_table()
        # self.get_target_sphere()

        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self.robots_to_log.append(self._frankas)
        # self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")
        self._drills = RigidPrimView(prim_paths_expr="/World/envs/.*/drill", name="drill_view", reset_xform_properties=False)
        self._tables = GeometryPrimView(prim_paths_expr="/World/envs/.*/table", name="cube_view", 
                                       reset_xform_properties=False)
        # self._target_spheres = XFormPrimView(prim_paths_expr="/World/envs/.*/target_sphere", name="target_view", 
        #                         reset_xform_properties=False)

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._thumb_fingers)
        scene.add(self._frankas._index_fingers)
        scene.add(self._drills)
        scene.add(self._tables)
        # scene.add(self._target_spheres)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._palm_centers)
        scene.add(self._frankas._thumb_fingers)
        scene.add(self._frankas._index_fingers)
        scene.add(self._drills)

        self.init_data()

    def get_franka(self):
        usd_path=f'{omniisaacgymenvs.__path__[0]}/models/franka_tekken.usd'
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", usd_path=usd_path)
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )

    def get_cabinet(self):
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet")
        self._sim_config.apply_articulation_settings(
            "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        )

    def get_drill(self):
        self._drill_position = torch.tensor([0.4, 0, 0.53], device=self._device)
        orientation = torch.tensor([0, 0, torch.pi], device=self._device).unsqueeze(0)
        self._drill_lower_bound = torch.tensor([0.3, -0.5, 0.53], device=self._device)
        self._drill_reset_lower_bound = torch.tensor([0.3, -0.5, 0.50], device=self._device)
        self._drill_upper_bound = torch.tensor([0.8, 0.5, 0.53], device=self._device)
        self._drills_rot = euler_angles_to_quats(orientation, device=self._device).squeeze(0)

        self._drill = Drill(prim_path=self.default_zero_env_path + '/drill',
                              name="drill",
                              position=self._drill_position,
                              orientation=self._drills_rot
                              )
        self._sim_config.apply_articulation_settings("drill", get_prim_at_path(self._drill.prim_path), self._sim_config.parse_actor_config("drill"))

    def get_table(self):
        self.cube_position = torch.tensor([0.2, 0., 0.2])
        self.cube_dimension = torch.tensor([0.6, 1, 0.4])
        self.cube_color = torch.tensor([0.22, 0.22, 0.22])
        table = FixedCuboid(prim_path= self.default_zero_env_path + "/table",
                                  name="table",
                                  translation= self.cube_position,
                                  scale = self.cube_dimension,
                                  color=self.cube_color,
                                  )
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    
    def get_target_sphere(self):
        self._target_sphere_color = torch.tensor([0.1, 0.9, 0.1], device=self._device)
        self._target_sphere_position = torch.tensor([0.6, 0., 0.7], device=self._device)
        self._target_sphere_lower_bound = torch.tensor([0.3, -0.5, 0.5], device=self._device)
        self._target_sphere_upper_bound = torch.tensor([0.9, 0.5, 1.], device=self._device)

        self._target_sphere = VisualSphere(prim_path= self.default_zero_env_path + "/target_sphere",
                                  name="target_sphere",
                                  translation= self._target_sphere_position,
                                  radius=0.05,
                                  color=self._target_sphere_color)
        
        self._target_sphere.set_collision_enabled(False)
        self._sim_config.apply_articulation_settings("target_sphere", get_prim_at_path(self._target_sphere.prim_path), self._sim_config.parse_actor_config("target_sphere"))


    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        palm_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/tekken/palm_link_hithand")),
            self._device,
        )

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, palm_pose[3:7], palm_pose[0:3]
        )
        franka_local_pose_pos += torch.tensor([0., 0., 0.02], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        drill_local_grasp_pose = torch.tensor([0.0045, 0.0, 0.026, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drill_local_grasp_pos = drill_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drill_local_grasp_rot = drill_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drill_inward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drill_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.drill_right_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.world_forward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.world_right_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [0, -0.99, 0., -2.6, -0., 3.14, 0.17] + [0.] * 20, device=self._device
        )

        self.joint_actions = torch.zeros((self._num_envs, 12), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        self.drill_pos, self.drill_rot = self._drills.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
            self.drill_grasp_rot,
            self.drill_grasp_pos,
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            self.drill_rot,
            self.drill_pos,
            self.drill_local_grasp_rot,
            self.drill_local_grasp_pos,
        )

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        to_target = self.drill_grasp_pos - self.franka_grasp_pos

        self.franka_thumb_pos, self.franka_thumb_rot = self._frankas._thumb_fingers.get_world_poses(clone=False)
        self.franka_index_pos, self.franka_index_rot = self._frankas._index_fingers.get_world_poses(clone=False)

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled[:, self._frankas.actuated_dof_indices],
                (franka_dof_vel * self.dof_vel_scale)[:, self._frankas.actuated_dof_indices],
                to_target,
                self.drill_pos[:, 2].unsqueeze(-1),
                self.drill_rot
            ),
            dim=-1,
        )

        self.compute_failure(self.drill_rot)
        # self._target_spheres.set_world_poses(positions=self.franka_grasp_pos, orientations=self.drill_grasp_rot)
        # print(self._frankas.get_measured_joint_efforts()[:, :7])
        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

        

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        self.joint_actions = self.actions

        # jeef = self._frankas.get_jacobians()[:, 6, :, :7]
        # dpose = actions[:, :6].unsqueeze(-1) * self.dt * self.ik_velocity
        # self.joint_actions[:, :7] = 1. * self.control_ik(j_eef=jeef, dpose=dpose, num_envs=self._num_envs, num_dofs=7)
        # self.joint_actions[:, 7:] = actions[:, 6:]

        targets = self.franka_dof_targets[:, self._frankas.actuated_dof_indices] + self.franka_dof_speed_scales[self._frankas.actuated_dof_indices] * self.dt * self.joint_actions * self.action_scale
        self.franka_dof_targets[:, self._frankas.actuated_dof_indices] = tensor_clamp(targets, self.franka_dof_lower_limits[self._frankas.actuated_dof_indices], self.franka_dof_upper_limits[self._frankas.actuated_dof_indices])
        
        self.franka_dof_targets[:, self._frankas.clamped_dof_indices[:5]] = self.franka_dof_targets[:, self._frankas.clamp_drive_dof_indices]
        self.franka_dof_targets[:, self._frankas.clamped_dof_indices[5:]] = self.franka_dof_targets[:, self._frankas.clamp_drive_dof_indices]
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, self._frankas.actuated_dof_indices] = pos[:, self._frankas.actuated_dof_indices]
        self.franka_dof_targets[env_ids, :] = dof_pos
        self.franka_dof_pos[env_ids, :] = dof_pos

        # reset cabinet
        # self._cabinets.set_joint_positions(
        #     torch.zeros_like(self._cabinets.get_joint_positions(clone=False)[env_ids]), indices=indices
        # )
        # self._cabinets.set_joint_velocities(
        #     torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)[env_ids]), indices=indices
        # )

        # # reset props
        # if self.num_props > 0:
        #     self._props.set_world_poses(
        #         self.default_prop_pos[self.prop_indices[env_ids].flatten()],
        #         self.default_prop_rot[self.prop_indices[env_ids].flatten()],
        #         self.prop_indices[env_ids].flatten().to(torch.int32),
        #     )

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        pos = torch.ones((num_indices, 3), device=self._device) * self._drill_position + self._env_pos[indices, :]
        rot = torch.ones((num_indices, 4), device=self._device) * self._drills_rot
        vel = torch.zeros((num_indices, 6), device=self._device)
        self._drills.set_world_poses(positions=pos,
                                     orientations=rot,
                                    indices=indices)
        self._drills.set_velocities(vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.joint_actions,
            self.franka_grasp_pos,
            self.drill_grasp_pos,
            self.drill_pos,
            self.franka_thumb_pos,
            self.franka_index_pos,
            self.franka_grasp_rot,
            self.drill_grasp_rot,
            self.gripper_forward_axis,
            self.drill_inward_axis,
            self.gripper_up_axis,
            self.drill_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self._max_episode_length,
            self.franka_dof_pos,
            self.finger_close_reward_scale,
            self.finger_open_reward_scale,
        )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(self.drill_pos[:, 2] > 0.6, torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(self.drill_pos[:, 2] <= self._drill_reset_lower_bound[2], torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.failed_envs, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos

    def compute_franka_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        franka_grasp_pos,
        drill_grasp_pos,
        drill_pos,
        franka_thumb_pos,
        franka_index_pos,
        franka_grasp_rot,
        drill_grasp_rot,
        gripper_forward_axis,
        drill_inward_axis,
        gripper_up_axis,
        drill_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        joint_positions,
        finger_close_reward_scale,
        finger_open_reward_scale,
    ):

        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drill_grasp_pos, p=2, dim=-1)
        dist_reward = torch.log(1 / (1 + d**2))
        dist_reward = torch.where(d <= 0.03, dist_reward + 0.5, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drill_grasp_rot, drill_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drill_grasp_rot, drill_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            franka_index_pos[:, 1] > drill_grasp_pos[:, 1],
            torch.where(
                franka_thumb_pos[:, 1] < drill_grasp_pos[:, 1], around_handle_reward + 0.5, around_handle_reward
            ),
            around_handle_reward,
        )

        # # reward for distance of each finger from the drawer
        # finger_dist_reward = torch.zeros_like(rot_reward)
        # lfinger_dist = torch.abs(franka_lfinger_pos[:, 1] - drill_grasp_pos[:, 1])
        # rfinger_dist = torch.abs(franka_rfinger_pos[:, 1] - drill_grasp_pos[:, 1])
        # finger_dist_reward = torch.where(
        #     franka_lfinger_pos[:, 1] > drill_grasp_pos[:, 1],
        #     torch.where(franka_rfinger_pos[:, 1] < drill_grasp_pos[:, 1], (0.04 - lfinger_dist) + (0.04 - rfinger_dist), 
        #                 finger_dist_reward),
        #     finger_dist_reward)
        

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.04, torch.sum(joint_positions[:, self._frankas.clamp_drive_dof_indices], dim=1), finger_close_reward
        )

        finger_open_reward = torch.zeros_like(rot_reward)
        finger_open_reward = torch.where(
            d >= 0.04, -torch.sum(joint_positions[:, self._frankas.clamp_drive_dof_indices], dim=1), finger_open_reward
        )


        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = torch.zeros_like(rot_reward)
        open_reward = torch.where(d <= 0.04, 1 / (1 + (0.6 - drill_pos[:, 2]) **2 ), open_reward)

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            + open_reward_scale * open_reward
            # + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            + finger_close_reward * finger_close_reward_scale
            + finger_open_reward * finger_open_reward_scale
        )

        # self.reward_terms_log["distReward"] = dist_reward
        # self.reward_terms_log["rotReward"] = rot_reward
        # self.reward_terms_log["aroundHandleReward"] = around_handle_reward
        # self.reward_terms_log["openReward"] = open_reward
        # self.reward_terms_log["fingerDistReward"] = finger_dist_reward
        # self.reward_terms_log["actionPenalty"] = action_penalty
        # self.reward_terms_log["fingerCloseReward"] = finger_close_reward


        # bonus for opening drawer properly
        rewards = torch.where(drill_pos[:, 2] > 0.56, rewards + 10 , rewards)
        rewards = torch.where(drill_pos[:, 2] > 0.6, rewards + 2 * 10, rewards)
        rewards = torch.where(drill_pos[:, 2] <= self._drill_reset_lower_bound[2], rewards - self.fail_penalty, rewards)
        rewards = torch.where(self.failed_envs, 
                                      rewards - self.fail_penalty,
                                      rewards)


        # # prevent bad style in opening drawer
        # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards
    
    def get_extras(self):
        pass
        # self.extras["success"] = self.drill_pos[:, 2] > 0.6
        # self.extras["rew_terms"] = self.reward_terms_log
        # self.extras["rew_weights"] = self.reward_weights_log
    
    def control_ik(self, j_eef, dpose, num_envs, num_dofs, damping=0.05):
        """Solve with Gauss Newton approx and regularizationin Isaac Gym.

        Returns: Change in DOF positions, [num_envs, num_dofs], to add to current positions.
        """
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(7).to(j_eef_T.device) * (damping ** 2)
        B = j_eef_T @ j_eef + lmbda
        g = j_eef_T @ dpose
        u = (torch.inverse(B) @ g).view(num_envs, num_dofs)
        return u
    
    def compute_failure(self, drill_rot, FAIL=0.6448):
        axis1 = tf_vector(drill_rot, self.drill_inward_axis)
        axis2 = tf_vector(drill_rot, self.drill_right_axis)


        dot1 = torch.abs(
            torch.bmm(axis1.view(self.num_envs, 1, 3), self.world_forward_axis.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of drill with world x
        dot2 = torch.abs(
            torch.bmm(axis2.view(self.num_envs, 1, 3), self.world_right_axis.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of drill with world y

        self.failed_envs = torch.logical_or(dot1 < FAIL, dot2 < FAIL)
