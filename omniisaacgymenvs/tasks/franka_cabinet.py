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

        self._num_actions = 12
        self.height_positioning_thr = 0.6
        self.rot_success_thr = 0.2
        self.pos_success_thr = 0.07

        self.show_target = False

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

        self.drill_alignment_reward_scale = self._task_cfg["env"]["drillAlignmentScale"]
        self.reward_weights_log["drillAlignmentReward"] = self._task_cfg["env"]["drillAlignmentScale"]

        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.reward_weights_log["openReward"] = self._task_cfg["env"]["openRewardScale"]

        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.reward_weights_log["actionPenalty"] = self._task_cfg["env"]["actionPenaltyScale"]

        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]
        self.reward_weights_log["fingerCloseReward"] = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.height_reward_scale = self._task_cfg["env"]["heightRewardScale"]
        self.reward_weights_log["heightReward"] = self._task_cfg["env"]["heightRewardScale"]

        self.fail_penalty = self._task_cfg["env"]["failPenalty"]

        self.success_type = self._task_cfg["env"]["successType"]

        self.goal_achieved_bonus = self._task_cfg["env"]["goalAchievedBonus"]
        self.reward_weights_log["goalBonusReward"] = 1.

        self.obs_type = self._task_cfg["env"]["obsType"]
        if self.obs_type =="full":
            self._num_observations = 70
        elif self.obs_type == "partial":
            self._num_observations = 37

        self.finger_reward_type = self._task_cfg["env"]["fingerRewardType"]
        self.pull_drill_enable = self._task_cfg["env"]["pullDrillEnable"]
        self.target_random_scaling = self._task_cfg["env"]["targetRandomScaling"]
        self.d_threshold = self._task_cfg["env"]["distanceThreshold"]
        

    def set_up_scene(self, scene) -> None:
        self.get_franka()
        self.get_drill()
        self.get_table()
        if self.show_target: self.get_target_sphere()

        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self.robots_to_log.append(self._frankas)
        # self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")
        if self.finger_reward_type == "enforce_contacts":
            self._drills = RigidPrimView(prim_paths_expr="/World/envs/.*/drill", name="drill_view", reset_xform_properties=False,
                                        prepare_contact_sensors=True,
                                        #    track_contact_forces=True,
                                        contact_filter_prim_paths_expr=["/World/envs/.*/franka/Right_Thumb_Phaprox",
                                                                        "/World/envs/.*/franka/Right_Thumb_Phamed",
                                                                        "/World/envs/.*/franka/Right_Thumb_Phadist",

                                                                        "/World/envs/.*/franka/Right_Index_Phaprox",
                                                                        "/World/envs/.*/franka/Right_Index_Phamed",
                                                                        "/World/envs/.*/franka/Right_Index_Phadist",

                                                                        "/World/envs/.*/franka/Right_Middle_Phaprox",
                                                                        "/World/envs/.*/franka/Right_Middle_Phamed",
                                                                        "/World/envs/.*/franka/Right_Middle_Phadist",

                                                                        "/World/envs/.*/franka/Right_Ring_Phaprox",
                                                                        "/World/envs/.*/franka/Right_Ring_Phamed",
                                                                        "/World/envs/.*/franka/Right_Ring_Phadist",

                                                                        "/World/envs/.*/franka/Right_Little_Phaprox",
                                                                        "/World/envs/.*/franka/Right_Little_Phamed",
                                                                        "/World/envs/.*/franka/Right_Little_Phadist",
                                                                        ]
                                            )
        else:
            self._drills = RigidPrimView(prim_paths_expr="/World/envs/.*/drill", name="drill_view", reset_xform_properties=False)

        self._tables = GeometryPrimView(prim_paths_expr="/World/envs/.*/table", name="cube_view", 
                                       reset_xform_properties=False)
        if self.show_target: self._target_spheres = XFormPrimView(prim_paths_expr="/World/envs/.*/target_sphere", name="target_view", 
                                reset_xform_properties=False)
        self._index_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/index_target", name="index_target_view", 
                                reset_xform_properties=False)
        self._middle_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/middle_target", name="middle_ringt_view", 
                                reset_xform_properties=False)
        self._ring_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/ring_target", name="ring_target_view", 
                                reset_xform_properties=False)
        self._little_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/little_target", name="little_target_view", 
                                reset_xform_properties=False)
        self._thumb_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/thumb_target", name="thumb_target_view", 
                                reset_xform_properties=False)

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._thumb_fingers)
        scene.add(self._frankas._index_fingers)
        scene.add(self._frankas._middle_fingers)
        scene.add(self._frankas._ring_fingers)
        scene.add(self._frankas._little_fingers)
        scene.add(self._drills)
        scene.add(self._index_targets)
        scene.add(self._middle_targets)
        scene.add(self._ring_targets)
        scene.add(self._little_targets)
        scene.add(self._thumb_targets)
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
        usd_path=f'{omniisaacgymenvs.__path__[0]}/models/franka_tekken_instantiable/franka_tekken_instantiable/franka_tekken_instantiable.usd'
        self.franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", usd_path=usd_path)
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(self.franka.prim_path), self._sim_config.parse_actor_config("franka")
        )

    def get_cabinet(self):
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet")
        self._sim_config.apply_articulation_settings(
            "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        )

    def get_drill(self):
        self._drill_position = torch.tensor([0.35, 0, 0.53], device=self._device)
        orientation = torch.tensor([0, 0, torch.pi], device=self._device).unsqueeze(0)
        self._drill_lower_bound = torch.tensor([0.25, -0.4, 0.53], device=self._device)
        self._drill_upper_bound = torch.tensor([0.6, 0.4, 0.53], device=self._device)
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
                                  radius=self.pos_success_thr,
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
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/palm_link_hithand")),
            self._device,
        )

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, palm_pose[3:7], palm_pose[0:3]
        )
        franka_local_pose_pos += torch.tensor([0.02, 0., 0.02], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        drill_local_grasp_pose = torch.tensor([0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0], device=self._device)
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

        self.world_forward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.world_right_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [0, -0.99, 0., -2.6, -0., 3.14, 0.17] + [0.] * 20, device=self._device
        )

        self.drill_target_pos = torch.tensor([0.5, 0, 0.75], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.drill_target_center = torch.tensor([0.3, 0., 0.75], device=self._device)
        self.drill_target_lower_bound = torch.tensor([0.1, -0.5, 0.7], device=self._device)
        self.drill_target_upper_bound = torch.tensor([0.6, 0.5, 0.85], device = self._device)


        self.default_drill_pos = torch.tensor([0.35, 0, 0.53], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.default_drill_rot = self._drills_rot.repeat((self._num_envs, 1))

        self.indexes_pos_target, _ = self._index_targets.get_local_poses()
        self.middles_pos_target, _ = self._middle_targets.get_local_poses()
        self.rings_pos_target, _ = self._ring_targets.get_local_poses()
        self.littles_pos_target, _ = self._little_targets.get_local_poses()
        self.thumbs_pos_target, _ = self._thumb_targets.get_local_poses()
        self.target_fingers_rotations = torch.tensor([ 1.0, 0.0, 0.0, 0.0], device = self._device).repeat(self._num_envs, 1)

        self.drill_pos = torch.ones((self._num_envs, 3), device=self._device) * self._drill_position + self._env_pos
        self.joint_actions = torch.zeros((self._num_envs, 12), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)

        self.indexes_pos, _ = self._frankas._index_fingers.get_world_poses(clone=False)
        self.middles_pos, _ = self._frankas._middle_fingers.get_world_poses(clone=False)
        self.rings_pos, _ = self._frankas._ring_fingers.get_world_poses(clone=False)
        self.littles_pos, _ = self._frankas._little_fingers.get_world_poses(clone=False)
        self.thumbs_pos, _ = self._frankas._thumb_fingers.get_world_poses(clone=False)

        self.drill_pos, self.drill_rot = self._drills.get_world_poses(clone=False)
        self.drill_vel = self._drills.get_velocities(clone=False)
        self.drill_linvel = self.drill_vel[:, 0:3]
        self.drill_angvel = self.drill_vel[:, 3:]
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

        to_drill = self.drill_grasp_pos - self.franka_grasp_pos
        to_target = self.drill_target_pos - (self.drill_pos - self._env_pos)

        self.d_target = torch.norm(to_target, p=2, dim=-1)
        self.d_default = torch.norm(self.drill_target_pos - (self.default_drill_pos - self._env_pos), p=2, dim=-1)
        quat_diff = quat_mul(self.drill_rot, quat_conjugate(self.default_drill_rot))
        self.target_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) 

        to_index = self.compute_finger_target_transforms(self.drill_rot,
                                                         self.drill_pos,
                                                         self.target_fingers_rotations,
                                                         self.indexes_pos_target) - self.indexes_pos
        self.d_index = torch.norm(to_index, p=2, dim=-1)
        to_middle = self.compute_finger_target_transforms(self.drill_rot,
                                                         self.drill_pos,
                                                         self.target_fingers_rotations,
                                                         self.middles_pos_target) - self.middles_pos
        self.d_middle = torch.norm(to_middle, p=2, dim=-1)
        to_ring = self.compute_finger_target_transforms(self.drill_rot,
                                                         self.drill_pos,
                                                         self.target_fingers_rotations,
                                                         self.rings_pos_target) - self.rings_pos
        self.d_ring = torch.norm(to_ring, p=2, dim=-1)
        to_little = self.compute_finger_target_transforms(self.drill_rot,
                                                         self.drill_pos,
                                                         self.target_fingers_rotations,
                                                         self.littles_pos_target) - self.littles_pos
        self.d_little = torch.norm(to_little, p=2, dim=-1)
        to_thumb = self.compute_finger_target_transforms(self.drill_rot,
                                                         self.drill_pos,
                                                         self.target_fingers_rotations,
                                                         self.thumbs_pos_target) - self.thumbs_pos
        self.d_thumb = torch.norm(to_thumb, p=2, dim=-1)


        self.franka_thumb_pos, self.franka_thumb_rot = self._frankas._thumb_fingers.get_world_poses(clone=False)
        self.franka_index_pos, self.franka_index_rot = self._frankas._index_fingers.get_world_poses(clone=False)
        if self.obs_type == "full":
            self.obs_buf = torch.cat(
                (
                    dof_pos_scaled[:, self._frankas.actuated_dof_indices],
                    (franka_dof_vel * self.dof_vel_scale)[:, self._frankas.actuated_dof_indices],
                    to_drill,
                    to_target,
                    self.drill_pos - self._env_pos,
                    self.drill_rot,
                    self.drill_linvel,
                    self.drill_angvel * 0.2,
                    self._frankas.get_measured_joint_efforts()[:, self._frankas.actuated_dof_indices],
                    self.indexes_pos - self._env_pos,
                    self.middles_pos - self._env_pos,
                    self.rings_pos - self._env_pos,
                    self.littles_pos - self._env_pos,
                    self.thumbs_pos - self._env_pos
                ),
                dim=-1,
            )
        elif self.obs_type == "partial":
            self.obs_buf = torch.cat(
                (
                    dof_pos_scaled[:, self._frankas.actuated_dof_indices],
                    (franka_dof_vel * self.dof_vel_scale)[:, self._frankas.actuated_dof_indices],
                    to_drill,
                    to_target,
                    self.drill_pos - self._env_pos,
                    self.drill_rot,
                ),
                dim=-1,
            )

        self.compute_failure(hand_pos, self.drill_rot)
        self.compute_success(self.success_type)
        if self.show_target: self._target_spheres.set_world_poses(positions=self.drill_target_pos + self._env_pos)
        # print(self._frankas.get_measured_joint_efforts()[:, :7])
        # print(self._frankas._middle_fingers.get_net_contact_forces())
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

        targets = self.franka_dof_targets[:, self._frankas.actuated_dof_indices] + self.franka_dof_speed_scales[self._frankas.actuated_dof_indices] * self.dt * self.joint_actions * self.action_scale
        self.franka_dof_targets[:, self._frankas.actuated_dof_indices] = tensor_clamp(targets, self.franka_dof_lower_limits[self._frankas.actuated_dof_indices], self.franka_dof_upper_limits[self._frankas.actuated_dof_indices])
        
        self.franka_dof_targets[:, self._frankas.clamped_dof_indices[:5]] = self.franka_dof_targets[:, self._frankas.clamp_drive_dof_indices]
        self.franka_dof_targets[:, self._frankas.clamped_dof_indices[5:]] = self.franka_dof_targets[:, self._frankas.clamp_drive_dof_indices]
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        if self.pull_drill_enable: self.pull_downward(strength=3.)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def pull_downward(self, strength=1.5):
        self._drills_to_pull = torch.where(self.drill_pos[:, 2] > 0.55, torch.ones_like(self._drills_to_pull), self._drills_to_pull)
        self.pull_env_ids = self._drills_to_pull.nonzero(as_tuple=False).squeeze(-1)

        if len(self.pull_env_ids) > 0:
            indices = self.pull_env_ids.to(dtype=torch.int32)
            self._drills.apply_forces_and_torques_at_pos(forces=torch.rand(3) * strength,
                                                         torques=torch.rand(3) * strength,
                                                        indices=indices)
        self._drills_to_pull[self.pull_env_ids] = 0

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



        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        pos = tensor_clamp(
            self._drill_position.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), 3), device=self._device) - 0.5),
            self._drill_lower_bound,
            self._drill_upper_bound,
            )

        dof_pos = torch.zeros((num_indices, 3), device=self._device)
        dof_pos[:, :] = pos + self._env_pos[env_ids]

        self.drill_target_pos[env_ids, :2] = pos[:, :2]
        # randomize yaw
        rot = torch.zeros((num_indices, 3), device=self._device)
        yaw = (torch.rand((len(env_ids),1), device=self._device)*2 - 1.) * 90 * torch.pi / 180 
        rot[:, 2] = (yaw + torch.pi).squeeze(-1)
        drill_rot = euler_angles_to_quats(rot, device=self._device)

        # rot = torch.ones((num_indices, 4), device=self._device) * self._drills_rot
        vel = torch.zeros((num_indices, 6), device=self._device)
        self._drills.set_world_poses(positions=dof_pos,
                                     orientations=drill_rot,
                                    indices=indices)
        self.default_drill_pos[env_ids, :] = dof_pos

        self._drills.set_velocities(vel, indices=indices)

        target_pos = tensor_clamp(
            self.drill_target_center.unsqueeze(0)
            + self.target_random_scaling * (torch.rand((len(env_ids), 3), device=self._device) - 0.5),
            self.drill_target_lower_bound,
            self.drill_target_upper_bound,
            )
        self.drill_target_pos[env_ids, :] = target_pos
        

        if hasattr(self, "_ref_cubes"):
            ref_cube_pos = dof_pos
            q = euler_angles_to_quats(torch.tensor([0, -torch.pi/2, 0], device=self._device).unsqueeze(0))
            rot = torch.ones((num_indices, 4), device=self._device) * q
            qx = euler_angles_to_quats(torch.tensor([0, 0, yaw], device=self._device).unsqueeze(0))
            rot = quat_mul(qx, rot)


            delta_pos = torch.tensor([-0.3, -0.06, 0.], device=self._device).repeat(num_indices,1)

            # Initialize quaternion representation of relative position
            p = torch.zeros((delta_pos.shape[0], 4), device=self._device)
            p[:, 1:4] = delta_pos

            # Convert relative position to the local frame
            q1I = quat_conjugate(drill_rot)
            p_prime = quat_mul(quat_mul(drill_rot, p), q1I)[:, 1:4]

        
            ref_cube_pos = p_prime + ref_cube_pos
            # ref_cube_pos[:, 2] = ref_cube_pos[:, 2] + torch.ones((num_indices, 1), device=self._device) *0.03

            self._ref_cubes.set_world_poses(positions=ref_cube_pos, orientations=rot, indices=indices)

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
        self.franka_dof_speed_scales[self._frankas.clamp_drive_dof_indices] = 1.5
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )
        self._drills_to_pull = torch.zeros(self.num_envs, device = self._device, dtype=torch.bool)

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
            self.open_reward_scale,
            self.height_reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self._max_episode_length,
            self.franka_dof_pos,
            self.finger_close_reward_scale,
        )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(self.drill_pos[:, 2] > 0.6, torch.ones_like(self.reset_buf), self.reset_buf)
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
    
    def compute_finger_target_transforms(
        self,
        drill_rot,
        drill_pos,
        drill_local_finger_target_rot,
        drill_local_finger_target_pos,

    ):
        _, global_finger_target_pos = tf_combine(
            drill_rot, drill_pos, drill_local_finger_target_rot, drill_local_finger_target_pos,
        )
        return (global_finger_target_pos)

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
        open_reward_scale,
        height_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        joint_positions,
        finger_close_reward_scale,
    ):

        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drill_grasp_pos, p=2, dim=-1)
        dist_reward = torch.log(1 / (1 + d**2))
        dist_reward = torch.where(d <= 0.03, dist_reward + 0.05, dist_reward)
        self.reward_terms_log["distReward"] = dist_reward

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

        # reward for matching the orientation of the hand to the drill (MAX 1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)
        self.reward_terms_log["rotReward"] = rot_reward

        finger_close_reward = torch.zeros_like(rot_reward)
        if self.finger_reward_type == "joint_position":
            # Rew for closing all the fingers joints (MAX 1)
            finger_close_reward = torch.where(
                d <= self.d_threshold, (1/15) * torch.sum(joint_positions[:, 12:], dim=1), finger_close_reward
            )
        elif(self.finger_reward_type == "fingertip_position"):
            # Rew for putting fingertip at target pos (MAX 1)
            finger_close_reward = torch.where(d <= self.d_threshold,
                                              0.2 * (1 / (1 + self.d_index**2) + 1 / (1 + self.d_middle**2) + 1 / (1 + self.d_ring**2) + 1 / (1 + self.d_little**2) + 1 / (1 + self.d_thumb**2)),
                                              finger_close_reward)
        elif(self.finger_reward_type == "enforce_contacts"):
            # Rew for maxing contacts with drill (MAX 1)
            cm = self._drills.get_contact_force_matrix()
            self.cm_bool_to_manipulability(cm)
            finger_close_reward = self.manipulability * 1/15

        else:
            print(f"Warning! invalid fingertp position reward type. Setting it to zero")
        self.reward_terms_log["fingerCloseReward"] = finger_close_reward
        
        # Reward for matching target orientation (MAX 1)
        drill_alignment_reward = torch.zeros_like(rot_reward)
        drill_alignment_reward = torch.where(self.drill_pos[:, 2] > self.height_positioning_thr,
                                              0.1 * (1.0 / (torch.abs(self.target_rot_dist) + 0.1)),
                                              drill_alignment_reward)
        self.reward_terms_log["drillAlignmentReward"] = drill_alignment_reward

        # regularization on the actions (summed for each environment) (MAX 1)
        action_penalty = 0.08 * torch.sum(actions**2, dim=-1)
        self.reward_terms_log["actionPenalty"] = action_penalty

        # how far the cabinet has been opened out (MAX 1)
        open_reward = torch.zeros_like(rot_reward)
        open_reward = torch.where(self.drill_pos[:, 2] > self.height_positioning_thr, (1 / (1 + self.d_target**2)), open_reward)
        self.reward_terms_log["openReward"] = open_reward

        # Bonus if it reaches the thr height (MAX 1)
        height_reward = torch.zeros_like(rot_reward)
        height_reward = torch.where(drill_pos[:, 2] > self.height_positioning_thr, height_reward + 1, height_reward)
        self.reward_terms_log["heightReward"] = height_reward

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + drill_alignment_reward * self.drill_alignment_reward_scale
            + open_reward_scale * open_reward
            + height_reward * height_reward_scale
            - action_penalty_scale * action_penalty
            + finger_close_reward * finger_close_reward_scale
        )

        # bonus for opening drawer properly
        if self.success_type == "positioning":
            rewards = torch.where(self.drill_pos[:, 2] > 0.7, rewards + self.goal_achieved_bonus, rewards)
            rewards = torch.where(self.success_envs, rewards + 2 * self.goal_achieved_bonus, rewards)

        if self.success_type == "positioning_orient":
            rewards = torch.where(self.success_rot_envs, rewards + self.goal_achieved_bonus, rewards)
            rewards = torch.where(self.success_rot_envs, rewards + self.goal_achieved_bonus, rewards)
            rewards = torch.where(self.success_envs, rewards + 4 * self.goal_achieved_bonus, rewards)
        self.reward_terms_log["goalBonusReward"] = torch.where(self.success_envs, self.goal_achieved_bonus, torch.zeros_like(rewards))
        
        rewards = torch.where(self.failed_envs, 
                                      rewards - self.fail_penalty,
                                      rewards)

        return rewards
    
    def get_extras(self):
        self.extras["success"] = self.success_envs
        if self.success_type == "positioning_orient":
            self.extras["success_pos"] = self.success_pos_envs
            self.extras["success_rot"] = self.success_rot_envs

        self.extras["rew_terms"] = self.reward_terms_log
        self.extras["rew_weights"] = self.reward_weights_log
    
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
    
    def compute_failure(self, hand_pos, drill_rot, RP_FAIL=0.5, YAW_FAIL=0.17):
        cos_roll = torch.abs(torch.cos(get_euler_xyz(drill_rot)[0]))
        cos_pitch = torch.abs(torch.cos(get_euler_xyz(drill_rot)[1]))

        self.failed_envs = torch.logical_or(hand_pos[:, 2] < 0.4, torch.logical_or(cos_roll < RP_FAIL, cos_pitch < RP_FAIL))


    def compute_success(self, success_type):
        if success_type == "lift":
            self.success_envs = self.drill_pos[:, 2] > self.height_positioning_thr
        if success_type == "positioning":
            self.success_envs = self.d_target <= self.pos_success_thr
        if success_type == "positioning_orient":
            self.success_envs = torch.logical_and(self.d_target <= self.pos_success_thr, torch.abs(self.target_rot_dist) <= self.rot_success_thr)
            self.success_pos_envs = self.d_target <= self.pos_success_thr
            self.success_rot_envs = torch.logical_and(self.drill_pos[:, 2] > self.height_positioning_thr, torch.abs(self.target_rot_dist) <= self.rot_success_thr)

    def cm_bool_to_manipulability(self, cm, TOL=1e-3):
        thumb_contact_idxs = [0, 1, 2]
        res = torch.norm(cm, dim=2) > TOL
        self.manipulability = torch.where(torch.logical_and(torch.any(res[:, thumb_contact_idxs], dim=1), torch.any(res[:, 3:], dim=1)),
                                           torch.count_nonzero(res, dim=1), 0.)