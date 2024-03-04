import math

import numpy as np
import torch
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere
from omniisaacgymenvs.robots.articulations.trial_robot import FrankaSimple



class TrialTask(RLTask):

    def __init__(self, name: str, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.translation = torch.tensor([0.0, 0.0, 0.0])
        self.cube_position = torch.tensor([0.5, 0., 0.2])
        self.cube_dimension = torch.tensor([0.6, 1, 0.4])
        self.sphere_radius = 0.1
        self.sphere_color = torch.tensor([0.1, 0.9, 0.1])
        self.sphere_position = torch.tensor([0.5, 0., 0.8])
        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 14 + 3 + 3
        self._num_actions = 7

        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_franka()
        self.get_cube()
        self.get_target_sphere()

        super().set_up_scene(scene)
        self._frankas = ArticulationView(
            prim_paths_expr="/World/envs/.*/franka",
            name="franka_view",
            reset_xform_properties=False)
        self.robot_to_log = self._frankas # Robot that gets logged by the logger
        
        scene.add(self._frankas)  # add view to scene for initialization

        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_link7", name="hands_view", reset_xform_properties=False)
        scene.add(self._hands)  # add view to scene for initialization
        
        self._spheres = GeometryPrimView(prim_paths_expr="/World/envs/.*/sphere", name="sphere_view", reset_xform_properties=False)
        scene.add(self._spheres)

        self._cubes = GeometryPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=False)
        scene.add(self._cubes)

        

    def get_franka(self):
        franka = FrankaSimple(prim_path=self.default_zero_env_path + "/franka",
                              usd_path="C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/tasks/panda_arm.usd",
                              name="franka",
                              translation=self.translation)
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path),
            self._sim_config.parse_actor_config("franka"))
        
    def get_cube(self):
        cube = FixedCuboid(prim_path= self.default_zero_env_path + "/cube",
                                  name="cube",
                                  translation= self.cube_position,
                                  scale = self.cube_dimension)
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube"))

    def get_target_sphere(self):
        sphere = VisualSphere(prim_path= self.default_zero_env_path + "/sphere",
                                  name="sphere",
                                  translation= self.sphere_position,
                                  radius = self.sphere_radius,
                                  color=self.sphere_color)
        self._sim_config.apply_articulation_settings("sphere", get_prim_at_path(sphere.prim_path), self._sim_config.parse_actor_config("sphere"))


    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        if not self._env._world.is_playing():
            return

        self.actions = actions.clone().to(self._device)
        targets = actions * torch.pi
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def get_observations(self) -> dict:
        dof_pos = self._frankas.get_joint_positions(clone=False)
        dof_vel = self._frankas.get_joint_velocities(clone=False)
        hand_pos, _ = self._hands.get_local_poses()
        target_pos, _ = self._spheres.get_local_poses()

        self._hand_pos = hand_pos
        self._target_pos = target_pos

        self.obs_buf[:, :7] = dof_pos
        self.obs_buf[:, 7:10] = hand_pos
        self.obs_buf[:, 10:13] = target_pos
        self.obs_buf[:, 13:20] = dof_vel
        # implement logic to retrieve observation states
        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        d = torch.norm(self._hand_pos - self._target_pos, p=2, dim=1)
        reward = -d
        return reward

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        pass
