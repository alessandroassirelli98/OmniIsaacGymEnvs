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
from omni.isaac.core.utils.torch.rotations import get_euler_xyz, quat_diff_rad
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere
from omniisaacgymenvs.robots.articulations.diana_tekken import DianaTekken
from omniisaacgymenvs.robots.articulations.views.diana_tekken_view import DianaTekkenView
from omniisaacgymenvs.robots.articulations.utils.kinematic_solver import KinematicsSolver



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

        self.diana_tekken_translation = torch.tensor([0.0, -0.15, 0.])

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 60
        self._num_actions = 22

        RLTask.__init__(self, name, env)


    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_tekken(name="diana",
                        usd_path="C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/models/diana_tekken/diana_tekken.usd",
                        translation=self.diana_tekken_translation)
        self.get_cube()
        self.get_target_sphere()

        super().set_up_scene(scene)

        self.diana_tekkens = DianaTekkenView(prim_paths_expr="/World/envs/.*/diana", name="tekken_view")
        self.robots_to_log.append(self.diana_tekkens) # Robot that gets logged by the logger
        scene.add(self.diana_tekkens)  # add view to scene for initialization

        scene.add(self.diana_tekkens._palm_centers)
        scene.add(self.diana_tekkens._index_fingers)
        scene.add(self.diana_tekkens._middle_fingers)
        scene.add(self.diana_tekkens._ring_fingers)
        scene.add(self.diana_tekkens._little_fingers)
        scene.add(self.diana_tekkens._thumb_fingers)

        self._spheres = RigidPrimView(prim_paths_expr="/World/envs/.*/sphere", name="sphere_view", reset_xform_properties=False)
        scene.add(self._spheres)
        
        self._cubes = GeometryPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=False)
        scene.add(self._cubes)


        
    def get_tekken(self, name, usd_path, translation):
        diana_tekken = DianaTekken(prim_path=self.default_zero_env_path + '/' + name,
                              usd_path=usd_path,
                              name=name,
                              translation=translation)
        self._sim_config.apply_articulation_settings(name, get_prim_at_path(diana_tekken.prim_path), self._sim_config.parse_actor_config(name))

        
    def get_cube(self):
        self.translation = torch.tensor([0.0, 0.0, 0.0])
        self.cube_position = torch.tensor([0.5, 0., 0.2])
        self.cube_dimension = torch.tensor([0.6, 1, 0.4])
        self.cube_color = torch.tensor([0.22, 0.22, 0.22])
        cube = FixedCuboid(prim_path= self.default_zero_env_path + "/cube",
                                  name="cube",
                                  translation= self.cube_position,
                                  scale = self.cube_dimension,
                                  color=self.cube_color)
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube"))
    
    def get_target_sphere(self):
        self._sphere_radius = 0.05
        self._sphere_color = torch.tensor([0.1, 0.9, 0.1], device=self._device)
        self._sphere_position = torch.tensor([0.5, 0., 0.46], device=self._device)
        self._sphere_lower_bound = torch.tensor([0.2, -0.5, 0.46], device=self._device)
        self._sphere_upper_bound = torch.tensor([0.8, 0.5, 0.46], device=self._device)

        sphere = DynamicSphere(prim_path= self.default_zero_env_path + "/sphere",
                                  name="sphere",
                                  translation= self._sphere_position,
                                  radius = self._sphere_radius,
                                  color=self._sphere_color,
                                  mass = 0.03)
        
        # sphere.set_collision_enabled(False) # Disable collision as it is used as a target
        self._sim_config.apply_articulation_settings("sphere", get_prim_at_path(sphere.prim_path), self._sim_config.parse_actor_config("sphere"))

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.num_diana_tekken_dofs = self.diana_tekkens.num_dof
        self.actuated_dof_indices = self.diana_tekkens.actuated_dof_indices
        self.num_actuated_dofs = len(self.actuated_dof_indices)
        self.default_dof_pos = torch.tensor([0., -0.4,  0., 1.3, 0., -1.3, 0.] + [0.] * 20, device=self._device)
        pos = self.default_dof_pos.unsqueeze(0) * torch.ones((self._num_envs, self.num_diana_tekken_dofs), device=self._device)

        self.diana_tekken_dof_targets = pos

        dof_limits = self.diana_tekkens.get_dof_limits()
        self.diana_tekken_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.diana_tekken_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.diana_tekkens.set_joint_positions(pos)
        self.diana_tekkens.set_joint_velocities(torch.zeros((self.num_envs, self.num_diana_tekken_dofs), device=self._device))
        self.diana_tekkens.set_joint_position_targets(pos)

        self.target_pos = torch.ones((self._num_envs, 3), device=self._device) * self._sphere_position  - self._env_pos


        self.spheres_to_pull = torch.zeros(self.num_envs, device = self._device)
        self.applied_ext_forces = torch.tensor([5., 5., -5.], device=self._device)

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        if not self._env._world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # self.push_downward()

        self.actions = actions.clone().to(self._device)
        targets = self.diana_tekken_dof_targets[:, self.actuated_dof_indices] + self.dt * self.actions * self.action_scale
        self.diana_tekken_dof_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.diana_tekken_dof_lower_limits[self.actuated_dof_indices], self.diana_tekken_dof_upper_limits[self.actuated_dof_indices])
        env_ids_int32 = torch.arange(self.diana_tekkens.count, dtype=torch.int32, device=self._device)

        self.diana_tekkens.set_joint_position_targets(self.diana_tekken_dof_targets, indices=env_ids_int32)


        
    def push_downward(self):
        self.spheres_to_pull = torch.where(self.target_pos[:, 2] > 0.5, torch.ones_like(self.spheres_to_pull), self.spheres_to_pull)
        pull_env_ids = self.spheres_to_pull.nonzero(as_tuple=False).squeeze(-1)

        if len(pull_env_ids) > 0:
            indices = pull_env_ids.to(dtype=torch.int32)
            self._spheres.apply_forces(self.applied_ext_forces, indices=indices)

            self.spheres_to_pull[pull_env_ids] = 0.

    def get_observations(self) -> dict:
        dof_pos = self.diana_tekkens.get_joint_positions(clone=False)
        dof_vel = self.diana_tekkens.get_joint_velocities(clone=False)
        hand_pos_world,  self.hand_rot = self.diana_tekkens._palm_centers.get_world_poses(clone=False)
        index_pos_world, _ = self.diana_tekkens._index_fingers.get_world_poses(clone=False)
        middle_pos_world, _ = self.diana_tekkens._middle_fingers.get_world_poses(clone=False)
        ring_pos_world, _ = self.diana_tekkens._ring_fingers.get_world_poses(clone=False)
        little_pos_world, _ = self.diana_tekkens._little_fingers.get_world_poses(clone=False)
        thumb_pos_world, _ = self.diana_tekkens._thumb_fingers.get_world_poses(clone=False)

        target_pos_world, self.target_rot = self._spheres.get_world_poses(clone=False)

        self.hand_pos = hand_pos_world - self._env_pos
        self.target_pos = target_pos_world - self._env_pos

        self.index_pose = index_pos_world - self._env_pos
        self.middle_pose = middle_pos_world - self._env_pos
        self.ring_pose = ring_pos_world - self._env_pos
        self.little_pose = little_pos_world - self._env_pos
        self.thumb_pose = thumb_pos_world - self._env_pos

        self.obs_buf[:, :27] = dof_pos
        self.obs_buf[:, 27:30] = self.hand_pos
        self.obs_buf[:, 30:33] = self.target_pos
        self.obs_buf[:, 33:60] = dof_vel
        # # implement logic to retrieve observation states
        observations = {self.diana_tekkens.name: {"obs_buf": self.obs_buf}}
        return observations


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
        d = torch.norm(self.hand_pos - self.target_pos, p=2, dim=1)
        reward = torch.log(1 / (1.0 + d ** 2))



        # reward = torch.where(torch.norm(self.hand_pos - self.target_pos, p=2, dim=1) < 0.05, reward + 1, reward)
        reward = torch.where(self.target_pos[:, 2] > 0.5, reward + 10, reward)

        reward = torch.where(torch.any(self.target_pos[:, :2] >= self._sphere_upper_bound[:2], dim=1), reward - 10, reward)
        reward = torch.where(torch.any(self.target_pos[:, :2] <= self._sphere_lower_bound[:2], dim=1), reward - 10, reward)

        self.rew_buf[:] = reward
        # pass

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # reset if max episode length is exceeded
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(torch.any(self.target_pos[:, :2] >= self._sphere_upper_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(torch.any(self.target_pos[:, :2] <= self._sphere_lower_bound[:2], dim=1), torch.ones_like(self.reset_buf), self.reset_buf)
        

