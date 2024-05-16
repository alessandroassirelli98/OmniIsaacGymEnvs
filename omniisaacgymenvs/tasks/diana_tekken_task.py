import math
import os
import numpy as np
import torch
import carb
from gym import spaces
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView, XFormPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import get_euler_xyz, quat_diff_rad, euler_angles_to_quats, quat_conjugate, quat_mul, quat_diff_rad
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere, DynamicCuboid
from omniisaacgymenvs.robots.articulations.diana_tekken import DianaTekken
from omniisaacgymenvs.robots.articulations.drill import Drill
from omniisaacgymenvs.robots.articulations.views.diana_tekken_view import DianaTekkenView
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from pxr import Usd, UsdGeom
from omni.isaac.core.utils.stage import get_current_stage



class DianaTekkenTask(RLTask):

    def __init__(self, name: str, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.robots_to_log = []

        self._robot_translation = torch.tensor([0.0, -0.15, 0.])

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 75
        if not hasattr(self, '_num_actions'): self._num_actions = 8 # If the number of actions has been defined from a child


        RLTask.__init__(self, name, env)


    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_robot(name="diana",
                        translation=self._robot_translation)
        self.get_cube()
        self.get_drill()
        # self.get_target_sphere()

        super().set_up_scene(scene)

        self._robots = DianaTekkenView(prim_paths_expr="/World/envs/.*/diana", name="tekken_view")
        self.robots_to_log.append(self._robots) # Robot that gets logged by the logger
        scene.add(self._robots)  # add view to scene for initialization

        # scene.add(self._robots._tool_centers)
        scene.add(self._robots._palm_centers)
        scene.add(self._robots._index_fingers)
        scene.add(self._robots._middle_fingers)
        scene.add(self._robots._ring_fingers)
        scene.add(self._robots._little_fingers)
        scene.add(self._robots._thumb_fingers)
        
        self._cubes = GeometryPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", 
                                       reset_xform_properties=False, 
                                    #    collisions = torch.ones(self.num_envs, dtype=torch.bool),
                                    #    prepare_contact_sensors=True,
                                    #    track_contact_forces=True,
                                    #    contact_filter_prim_paths_expr=["/World/envs/.*/diana/.*"]
                                       )
        scene.add(self._cubes)

        self._drills = RigidPrimView(prim_paths_expr="/World/envs/.*/drill", name="drill_view", reset_xform_properties=False,
                                    #    prepare_contact_sensors=True,
                                    # #    track_contact_forces=True,
                                    #    contact_filter_prim_paths_expr=["/World/envs/.*/diana/Right_Thumb_Phaprox",
                                    #                                    "/World/envs/.*/diana/Right_Thumb_Phamed",
                                    #                                    "/World/envs/.*/diana/Right_Thumb_Phadist",

                                    #                                    "/World/envs/.*/diana/Right_Index_Phaprox",
                                    #                                    "/World/envs/.*/diana/Right_Index_Phamed",
                                    #                                    "/World/envs/.*/diana/Right_Index_Phadist",

                                    #                                    "/World/envs/.*/diana/Right_Middle_Phaprox",
                                    #                                    "/World/envs/.*/diana/Right_Middle_Phamed",
                                    #                                    "/World/envs/.*/diana/Right_Middle_Phadist",

                                    #                                    "/World/envs/.*/diana/Right_Ring_Phaprox",
                                    #                                    "/World/envs/.*/diana/Right_Ring_Phamed",
                                    #                                    "/World/envs/.*/diana/Right_Ring_Phadist",

                                    #                                    "/World/envs/.*/diana/Right_Little_Phaprox",
                                    #                                    "/World/envs/.*/diana/Right_Little_Phamed",
                                    #                                    "/World/envs/.*/diana/Right_Little_Phadist",
                                    #                                    ]
                                                                       )
        scene.add(self._drills)

        self._drills_finger_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/finger_target_pos", name="finger_targets", reset_xform_properties=False)
        scene.add(self._drills_finger_targets)


        # self._target_spheres = XFormPrimView(prim_paths_expr="/World/envs/.*/target_sphere", name="target_view", 
        #                         reset_xform_properties=False)
        # scene.add(self._target_spheres)


        
    def get_robot(self, name, translation):
        self._hand_lower_bound = torch.tensor([0.0, -0.5, 0.2], device=self._device)
        self._hand_upper_bound = torch.tensor([0.9, 0.5, 0.9], device=self._device)
        self._robot = DianaTekken(prim_path=self.default_zero_env_path + '/' + name,
                              name=name,
                              translation=translation)
        self._sim_config.apply_articulation_settings(name, get_prim_at_path(self._robot.prim_path), self._sim_config.parse_actor_config(name))

    def get_drill(self):
        self._drill_position = torch.tensor([0.6, 0, 0.53], device=self._device)
        orientation = torch.tensor([0, 0, 0], device=self._device).unsqueeze(0)
        self._drill_lower_bound = torch.tensor([0.3, -0.5, 0.53], device=self._device)
        self._drill_reset_lower_bound = torch.tensor([0.3, -0.5, 0.45], device=self._device)
        self._drill_upper_bound = torch.tensor([0.8, 0.5, 0.53], device=self._device)
        self._drills_rot = euler_angles_to_quats(orientation, device=self._device)

        self._drill = Drill(prim_path=self.default_zero_env_path + '/drill',
                              name="drill",
                              position=self._drill_position,
                              )
        self._sim_config.apply_articulation_settings("drill", get_prim_at_path(self._drill.prim_path), self._sim_config.parse_actor_config("drill"))
 
    def get_cube(self):
        self.cube_position = torch.tensor([0.6, 0., 0.2])
        self.cube_dimension = torch.tensor([0.6, 1, 0.4])
        self.cube_color = torch.tensor([0.22, 0.22, 0.22])
        cube = FixedCuboid(prim_path= self.default_zero_env_path + "/cube",
                                  name="cube",
                                  translation= self.cube_position,
                                  scale = self.cube_dimension,
                                  color=self.cube_color,
                                  )
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube"))

    def get_target_sphere(self):
        self._target_sphere_color = torch.tensor([0.1, 0.9, 0.1], device=self._device)
        self._target_sphere_position = torch.tensor([0.6, 0., 0.7], device=self._device)
        self._target_sphere_lower_bound = torch.tensor([0.3, -0.5, 0.5], device=self._device)
        self._target_sphere_upper_bound = torch.tensor([0.9, 0.5, 1.], device=self._device)

        self._target_sphere = VisualSphere(prim_path= self.default_zero_env_path + "/target_sphere",
                                  name="target_sphere",
                                  translation= self._target_sphere_position,
                                  radius=0.01,
                                  color=self._target_sphere_color)
        
        self._target_sphere.set_collision_enabled(False)
        self._sim_config.apply_articulation_settings("target_sphere", get_prim_at_path(self._target_sphere.prim_path), self._sim_config.parse_actor_config("target_sphere"))

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.manipulability = torch.zeros((self.num_envs), dtype=torch.bool, device = self._device)
        self.bool_contacts = torch.zeros((self.num_envs, 15), dtype=torch.bool, device=self._device)
    
        self.num_diana_tekken_dofs = self._robots.num_dof
        self.actuated_dof_indices = self._robots.actuated_dof_indices
        self.num_actuated_dofs = len(self.actuated_dof_indices)
        # 0.3311, -0.8079, -0.4242,  2.2495,  2.7821,  0.0904,  1.6300
        self.default_dof_pos = torch.tensor([0.3311, -0.8079, -0.4242,  2.2495,  2.7821,  0.0904,  1.6300]  + [0.] * 20, device=self._device)
        pos = self.default_dof_pos.unsqueeze(0) * torch.ones((self._num_envs, self.num_diana_tekken_dofs), device=self._device)

        self._robot_dof_targets = pos
        self._ref_joint_targets = torch.tensor([2.1851e-01,  3.3276e-02,  8.0705e-01,  4.9647e-01,  2.3982e-01,
                                                8.9000e-01,  1.1345e+00,  1.1345e+00,  6.9859e-01,  1.0972e+00], 
                                            device=self._device)
        self._ref_joint_targets = self._ref_joint_targets * torch.ones((self._num_envs, 10), device=self._device)

        self._ref_grasp_in_drill_pos = torch.tensor([-0.0269, -0.0307, -0.0138], 
                                            device=self._device)
        self._ref_grasp_in_drill_pos = self._ref_grasp_in_drill_pos * torch.ones((self._num_envs, 3), device=self._device)

        self._ref_grasp_in_drill_rot = torch.tensor([-0.9926,  0.1128,  0.0436, -0.0108], 
                                            device=self._device)
        self._ref_grasp_in_drill_rot = self._ref_grasp_in_drill_rot * torch.ones((self._num_envs, 4), device=self._device)

        drill_pos, _ = self._drills.get_world_poses()
        drill_finger_targets, _ = self._drills_finger_targets.get_world_poses()
        self.finger_target_offset = drill_finger_targets - drill_pos
        

        self.reach_target = torch.tensor([0.8, 0., 0.6], device=self._device)

        dof_limits = self._robots.get_dof_limits()
        self._robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self._robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self._robots.set_joint_positions(pos)
        self._robots.set_joint_velocities(torch.zeros((self.num_envs, self.num_diana_tekken_dofs), device=self._device))
        self._robots.set_joint_position_targets(pos)

        self.drill_pos = torch.ones((self._num_envs, 3), device=self._device) * self._drill_position  - self._env_pos
        self.drill_rot = torch.ones((self._num_envs, 4), device=self._device) * self._drills_rot
        
        self.drill_zero_rot = torch.ones((self._num_envs, 4), device=self._device) * self._drills_rot

        # self.target_sphere_pos = torch.ones((self._num_envs, 3), device=self._device) * self._target_sphere_position

        self._cubes_to_pull = torch.zeros(self.num_envs, device = self._device)
        self.applied_ext_forces = torch.tensor([1., 1., -1.], device=self._device)

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices, False)
        
    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        if not self._env._world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # pass
            self.reset_idx(reset_env_ids, False)

        self.actions = actions.clone().to(self._device)
        self._robot_dof_targets[:, self.actuated_dof_indices] += self.actions * self.dt * self.action_scale
        self._robot_dof_targets = self._robots.clamp_joint0_joint1(self._robot_dof_targets)

        self._robot_dof_targets[:, self.actuated_dof_indices] = tensor_clamp(self._robot_dof_targets[:, self.actuated_dof_indices], self._robot_dof_lower_limits[self.actuated_dof_indices], self._robot_dof_upper_limits[self.actuated_dof_indices])
        env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        self._robots.set_joint_position_targets(self._robot_dof_targets, indices=env_ids_int32)

        # print(self._robots.get_measured_joint_forces()[:, 12, 3])

        # print(self._robots.get_joint_positions()[:, 7:])

    # def push_downward(self):
    #     self._cubes_to_pull = torch.where(self.drill_pos[:, 2] > 0.6, torch.ones_like(self._cubes_to_pull), self._cubes_to_pull)
    #     pull_env_ids = self._cubes_to_pull.nonzero(as_tuple=False).squeeze(-1)

    #     if len(pull_env_ids) > 0:
    #         indices = pull_env_ids.to(dtype=torch.int32)
    #         self._pick_up_cubes.apply_forces(self.applied_ext_forces, indices=indices)

    #         self._cubes_to_pull[pull_env_ids] = 0.

    def get_observations(self) -> dict:
        def get_in_object_pose(p1, p2, q1, q2):
            """
            Compute pose in coordinate in local frame.

            Args:
                local_pos1 (torch.Tensor): Local position 1.
                local_pos2 (torch.Tensor): Local position 2.
                world_pose1 (torch.Tensor): World pose 1 as a quaternion.
                world_pose2 (torch.Tensor): World pose 2 as a quaternion.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Tuple containing the transformed
                position and quaternion representing the pose in the local frame.
            """
            # Compute relative position in the local frame
            delta_pos = p2 - p1

            # Initialize quaternion representation of relative position
            p = torch.zeros((p1.shape[0], 4), device=self._device)
            p[:, 1:4] = delta_pos

            # Convert relative position to the local frame
            q1I = quat_conjugate(q1)
            p_prime = quat_mul(quat_mul(q1I, p), q1)[:, 1:4]

            # Convert world_pose2 to the local frame
            q_prime = quat_mul(q1I, q2)

            return p_prime, q_prime
        

        self.dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)
        hand_pos_world,  self.hand_rot = self._robots._palm_centers.get_world_poses(clone=False)
        index_pos_world, _ = self._robots._index_fingers.get_world_poses(clone=False)
        middle_pos_world, _ = self._robots._middle_fingers.get_world_poses(clone=False)
        ring_pos_world, _ = self._robots._ring_fingers.get_world_poses(clone=False)
        little_pos_world, _ = self._robots._little_fingers.get_world_poses(clone=False)
        thumb_pos_world, _ = self._robots._thumb_fingers.get_world_poses(clone=False)

        drill_pos_world, self.drill_rot = self._drills.get_world_poses(clone=False)

        self.hand_pos = hand_pos_world - self._env_pos
        self.drill_pos = drill_pos_world - self._env_pos

        self.hand_in_drill_pos, self.hand_in_drill_rot = get_in_object_pose(self.drill_pos, self.hand_pos, self.drill_rot, self.hand_rot)
        self.drill_finger_targets_pos = drill_pos_world + self.finger_target_offset

        # Rotate the offset vector from local frame to world frame.
        # Then add the drill position to get the position of the target in world
        p = torch.zeros((self._num_envs, 4), device=self._device)
        p[:, 1:4] = self.finger_target_offset
        q = self.drill_rot
        qI = quat_conjugate(q)
        self.drill_finger_targets_pos = self.drill_pos + quat_mul(quat_mul(q, p), qI)[:, 1:4]
        # print(self.drill_finger_targets_pos)
        
        # print(f'pos: {self.hand_in_drill_pos} rot:{self.hand_in_drill_rot}')
        # print(f'Joints: {self.dof_pos[:, 7:]}')

        self.index_pos = index_pos_world - self._env_pos
        self.middle_pos = middle_pos_world - self._env_pos
        self.ring_pos = ring_pos_world - self._env_pos
        self.little_pos = little_pos_world - self._env_pos
        self.thumb_pos = thumb_pos_world - self._env_pos

        # self._target_spheres.set_world_poses(positions=self.drill_finger_targets_pos)


        self.obs_buf[:, :27] = self.dof_pos
        self.obs_buf[:, 27:30] = self.hand_pos
        self.obs_buf[:, 30:34] = self.hand_rot
        self.obs_buf[:, 34:37] = self.drill_pos
        self.obs_buf[:, 37:41] = self.drill_rot
        self.obs_buf[:, 41:44] = self.hand_in_drill_pos
        self.obs_buf[:, 44:48] = self.hand_in_drill_rot
        # self.obs_buf[:, 34:37] = self.target_sphere_pos
        self.obs_buf[:, 48:75] = dof_vel

        # self.obs_buf[:, 41:68] = dof_vel
        # # implement logic to retrieve observation states
        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations


    def reset_idx(self, env_ids, deterministic=False):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # Reset Diana Tekken robots
        if not deterministic:
            pos = tensor_clamp(
                self.default_dof_pos[self.actuated_dof_indices].unsqueeze(0)
                + 0.25 * (torch.rand((len(env_ids), self.num_actuated_dofs), device=self._device) - 0.5),
                self._robot_dof_lower_limits[self.actuated_dof_indices],
                self._robot_dof_upper_limits[self.actuated_dof_indices],
            )
        else:
            pos = self.default_dof_pos[self.actuated_dof_indices].unsqueeze(0)

        dof_pos = torch.zeros((num_indices, self.num_diana_tekken_dofs), device=self._device)
        dof_vel = torch.zeros((num_indices, self.num_diana_tekken_dofs), device=self._device)

        dof_pos[:, self.actuated_dof_indices] = pos
        self._robot_dof_targets[env_ids, :] = dof_pos

        
        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)
        self._robots.set_joint_position_targets(self._robot_dof_targets[env_ids], indices=indices)

        # Reset target positions
        # pos = tensor_clamp(
        #     self._target_sphere_position.unsqueeze(0)
        #     + 0.25 * (torch.rand((len(env_ids), 3), device=self._device) - 0.5),
        #     self._target_sphere_lower_bound,
        #     self._target_sphere_upper_bound,
        # )
        # dof_pos = torch.zeros((num_indices, 3), device=self._device)
        # dof_pos[:, :] = pos + self._env_pos[env_ids]
        # self._target_spheres.set_world_poses(positions=dof_pos, indices=indices)


        # Reset drill positions
        if not deterministic:
            pos = tensor_clamp(
                self._drill_position.unsqueeze(0)
                + 0.25 * (torch.rand((len(env_ids), 3), device=self._device) - 0.5),
                self._drill_lower_bound,
                self._drill_upper_bound,
            )
        else:
            pos = self._drill_position.unsqueeze(0)

        dof_pos = torch.zeros((num_indices, 3), device=self._device)
        dof_pos[:, :] = pos + self._env_pos[env_ids]

        
        rot = torch.ones((num_indices, 4), device=self._device) * self._drills_rot

        self._drills.set_velocities(torch.zeros((num_indices, 6)), indices=indices)
        self._drills.set_world_poses(positions=dof_pos, orientations=rot, indices=indices)

        if hasattr(self, "_ref_cubes"):
            ref_cube_pos = dof_pos
            q = euler_angles_to_quats(torch.tensor([torch.pi/2, 0, -torch.pi/2], device=self._device).unsqueeze(0))
            rot = torch.ones((num_indices, 4), device=self._device) * q

            ref_cube_pos[:, 0] = ref_cube_pos[:, 0] - torch.ones((num_indices, 1), device=self._device) * 0.4
            ref_cube_pos[:, 2] = ref_cube_pos[:, 2] + torch.ones((num_indices, 1), device=self._device) *0.01

            self._ref_cubes.set_world_poses(positions=ref_cube_pos, orientations=rot, indices=indices)


        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def calculate_metrics(self) -> None:
        reward = torch.zeros(self._num_envs, device=self._device)
        fail_penalty = 10
        goal_achieved = 1
        # implement logic to compute rewards

        # Distance hand to drill grasp pos
        d = torch.norm(self.hand_in_drill_pos - self._ref_grasp_in_drill_pos, p=2, dim=1)
        reward = self.add_reward_term(d, reward)
        reward = torch.where(torch.norm(self.hand_in_drill_pos - self._ref_grasp_in_drill_pos, p=2, dim=1) < 0.05, reward + 0.2, reward)

        # rotation difference
        d = quat_diff_rad(self.hand_in_drill_rot, self._ref_grasp_in_drill_rot)
        reward = self.add_reward_term(d, reward)
        # print(d)

        # Distance to target height
        d = torch.abs(0.7 - self.drill_pos[:, 2])
        reward = self.add_reward_term(d, reward, 0.5)

        # Orientation cost
        d = quat_diff_rad(self.drill_zero_rot, self.drill_rot)
        reward = self.add_reward_term(d, reward, 0.2)

        # Fingertip distance from reference
        reward = self.add_reward_term(torch.norm(self.drill_finger_targets_pos - self.index_pos, p=2, dim=1), reward, 0.05)
        reward = self.add_reward_term(torch.norm(self.drill_finger_targets_pos - self.middle_pos, p=2, dim=1), reward, 0.05)
        reward = self.add_reward_term(torch.norm(self.drill_finger_targets_pos - self.ring_pos, p=2, dim=1), reward, 0.05)
        reward = self.add_reward_term(torch.norm(self.drill_finger_targets_pos - self.little_pos, p=2, dim=1), reward, 0.05)
        reward = self.add_reward_term(torch.norm(self.drill_finger_targets_pos - self.thumb_pos, p=2, dim=1), reward, 0.05)

        # Prize if goal achieved
        reward = torch.where(self.drill_pos[:, 2] > 0.7, reward + goal_achieved, reward)

        # If the drill is out of bound
        reward = torch.where(torch.any(self.drill_pos[:, :2] >= self._drill_upper_bound[:2], dim=1), reward - fail_penalty, reward)
        reward = torch.where(torch.any(self.drill_pos <= self._drill_reset_lower_bound, dim=1), reward - fail_penalty, reward)

        # If the hand is out of bound
        reward = torch.where(torch.any(self.hand_pos[:, :2] >= self._hand_upper_bound[:2], dim=1), reward - fail_penalty, reward)
        reward = torch.where(torch.any(self.hand_pos[:, :2] <= self._hand_lower_bound[:2], dim=1), reward - fail_penalty, reward)
        # print(reward)
        

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # reset if max episode length is exceeded
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # If the drill is out of bound
        self.reset_buf = torch.where(torch.any(self.drill_pos[:, :2] >= self._drill_upper_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(torch.any(self.drill_pos <= self._drill_reset_lower_bound, dim=1), torch.ones_like(self.reset_buf), self.reset_buf)

        # # # If the hand is out of bound
        self.reset_buf = torch.where(torch.any(self.hand_pos[:, :2] >= self._hand_upper_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(torch.any(self.hand_pos[:, :2] <= self._hand_lower_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)

        # # # Task achieved
        # self.reset_buf = torch.where(self.drill_pos[:, 2] > 0.6, torch.ones_like(self.reset_buf), self.reset_buf)

        # If the resultant contact force between table and hand is more than a threshold
        # self.reset_buf = torch.where(torch.sqrt(torch.sum(self._cubes.get_contact_force_matrix()[:, 0, :]**2, dim=1 )) >= 0.5, 
        #                              torch.ones_like(self.reset_buf), self.reset_buf) #Doesn't work on gpu

    def cm_bool_to_manipulability(self, cm, TOL=1e-3):
        thumb_contact_idxs = [0, 1, 2]
        res = torch.norm(cm, dim=2) > TOL
        self.manipulability = torch.where(torch.logical_and(torch.any(res[:, thumb_contact_idxs], dim=1), torch.any(res[:, 3:], dim=1)),
                                           torch.count_nonzero(res, dim=1), 0.)

    def add_reward_term(self, d, reward, w=1):
        return reward + torch.log(1 / (1.0 + d ** 2)) * w