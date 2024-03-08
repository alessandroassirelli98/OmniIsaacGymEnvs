import math

import numpy as np
import torch
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import get_euler_xyz, quat_diff_rad
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere
from omniisaacgymenvs.robots.articulations.tekken import Tekken



class TekkenTask(RLTask):

    def __init__(self, name: str, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.translation = torch.tensor([0.0, 0.0, 0.8])
        self.cube_position = torch.tensor([0.5, 0., 0.2])
        self.cube_dimension = torch.tensor([0.6, 1, 0.4])
        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 14 + 3 + 3
        self._num_actions = 20

        RLTask.__init__(self, name, env)


    def set_up_scene(self, scene) -> None:
        self._sphere_radius = 0.1
        self._sphere_color = torch.tensor([0.1, 0.9, 0.1], device=self._device)
        self._sphere_position = torch.tensor([0.5, 0., 0.7], device=self._device)
        self._sphere_lower_bound = torch.tensor([0.2, -0.5, 0.5], device=self._device)
        self._sphere_upper_bound = torch.tensor([0.8, 0.5, 1.], device=self._device)

        # implement environment setup here
        self.get_tekken()
        self.get_cube()
        self.get_target_sphere()

        super().set_up_scene(scene)
        self._tekkens = ArticulationView(
            prim_paths_expr="/World/envs/.*/hithand",
            name="tekken_view",
            reset_xform_properties=False)
        self.robot_to_log = self._tekkens # Robot that gets logged by the logger

        scene.add(self._tekkens)  # add view to scene for initialization
        
        self._spheres = RigidPrimView(prim_paths_expr="/World/envs/.*/sphere", name="sphere_view", reset_xform_properties=False)
        scene.add(self._spheres)
        
        self._cubes = GeometryPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=False)
        scene.add(self._cubes)
        
    def get_tekken(self):
        hithand = Tekken(prim_path=self.default_zero_env_path + "/hithand",
                              usd_path="C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/models/tekken_cad/tekken_cad.usd",
                              name="hithand",
                              translation=self.translation)
        self._sim_config.apply_articulation_settings("hithand", get_prim_at_path(hithand.prim_path), self._sim_config.parse_actor_config("hithand"))
        
    def get_cube(self):
        cube = FixedCuboid(prim_path= self.default_zero_env_path + "/cube",
                                  name="cube",
                                  translation= self.cube_position,
                                  scale = self.cube_dimension)
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube"))

    def get_target_sphere(self):
        sphere = DynamicSphere(prim_path= self.default_zero_env_path + "/sphere",
                                  name="sphere",
                                  translation= self._sphere_position,
                                  radius = self._sphere_radius,
                                  color=self._sphere_color)
        
        sphere.set_collision_enabled(False) # Disable collision as it is used as a target
        self._sim_config.apply_articulation_settings("sphere", get_prim_at_path(sphere.prim_path), self._sim_config.parse_actor_config("sphere"))
        
    def post_reset(self):
        # implement any logic required for simulation on-start here
        pass
        
    def init_franka(self):
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        if not self._env._world.is_playing():
            return
        
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)

        # self.actions = actions.clone().to(self._device)
        # targets = self.franka_dof_targets + self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
        # self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def get_observations(self) -> dict:
        # dof_pos = self._frankas.get_joint_positions(clone=False)
        # dof_vel = self._frankas.get_joint_velocities(clone=False)
        # hand_pos_world,  self.hand_rot = self._hands.get_world_poses(clone=False)
        # target_pos_world, self.target_rot = self._spheres.get_world_poses(clone=False)

        # self.hand_pos = hand_pos_world - self._env_pos
        # self.target_pos = target_pos_world - self._env_pos

        # self.obs_buf[:, :7] = dof_pos
        # self.obs_buf[:, 7:10] = self.hand_pos
        # self.obs_buf[:, 10:13] = self.target_pos
        # self.obs_buf[:, 13:20] = dof_vel
        # # implement logic to retrieve observation states
        # observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        # return observations
        pass

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # Reset Franka robots
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)
        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)

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




        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        # Distance to target
        pass

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # reset if max episode length is exceeded
        pass

