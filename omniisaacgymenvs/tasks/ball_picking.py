import math

import numpy as np
import torch
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import get_euler_xyz, quat_diff_rad, euler_angles_to_quats
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere
from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView



class BallPicking(RLTask):

    def __init__(self, name: str, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.table_position = torch.tensor([0.5, 0., 0.2])
        self.table_dimension = torch.tensor([0.6, 1, 0.4])
        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 14 + 3 + 3
        self._num_actions = 20

        RLTask.__init__(self, name, env)


    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()

        # implement environment setup here
        self.get_hand()
        self.get_table()
        self.get_target_sphere()
        self.get_ball()

        super().set_up_scene(scene)

        self._shadow_hands = ShadowHandView(prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view")
        scene.add(self._shadow_hands)

        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view", reset_xform_properties=False)
        scene.add(self._balls)
        
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)
        
        self._tables = GeometryPrimView(prim_paths_expr="/World/envs/.*/table", name="table_view", reset_xform_properties=False)
        scene.add(self._tables)

        self.init_franka()

        
    def get_hand(self):
        self.start_hand_translation = torch.tensor([0.1, 0.0, 0.5], device=self._device)
        self.start_hand_orientation = euler_angles_to_quats(torch.tensor([torch.pi/2, 0., torch.pi/2]).unsqueeze(0)).squeeze(0)
        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand", 
            usd_path="C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/tasks/cartpole.usd",
            name="shadow_hand",
            translation=self.start_hand_translation, 
            orientation=self.start_hand_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "shadow_hand", 
            get_prim_at_path(shadow_hand.prim_path), 
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        shadow_hand.set_shadow_hand_properties(stage=self._stage, shadow_hand_prim=shadow_hand.prim)
        shadow_hand.set_motor_control_mode(stage=self._stage, shadow_hand_path=shadow_hand.prim_path)
        
    def get_table(self):
        table = FixedCuboid(prim_path= self.default_zero_env_path + "/table",
                                  name="table",
                                  translation= self.table_position,
                                  scale = self.table_dimension)
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def get_target_sphere(self):
        self._sphere_color = torch.tensor([0.1, 0.9, 0.1], device=self._device)
        self._target_position = torch.tensor([0.5, 0., 0.7], device=self._device)
        self._target_lower_bound = torch.tensor([0.2, -0.5, 0.5], device=self._device)
        self._target_upper_bound = torch.tensor([0.8, 0.5, 1.], device=self._device)
        target = DynamicSphere(prim_path= self.default_zero_env_path + "/target",
                                  name="target",
                                  translation= self._target_position,
                                  radius = 0.03,
                                  color=self._sphere_color)
        
        target.set_collision_enabled(False) # Disable collision as it is used as a target
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))

    def get_ball(self):
        self._ball_position = torch.tensor([0.5, 0., 0.45], device=self._device)
        self._ball_color =  torch.tensor([1., 0., 0.], device=self._device)
        target = DynamicSphere(prim_path= self.default_zero_env_path + "/ball",
                                  name="ball",
                                  translation= self._ball_position,
                                  radius = 0.03,
                                  color=self._ball_color)
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("ball"))
        
    def post_reset(self):
        self.targets = torch.zeros((self.num_envs, self._shadow_hands.num_dof), dtype=torch.float, device=self.device)

        # implement any logic required for simulation on-start here
        pass
        
    def init_franka(self):
        # self.franka_default_dof_pos = torch.zeros((self._num_actions), dtype=torch.float32, device=self._device)
        # self.actions = torch.zeros((self._num_envs, self.num_actions), dtype=torch.float32, device=self._device)
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        # if not self._env._world.is_playing():
        #     return
        
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)

        # self.actions = actions.clone().to(self._device)
        # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
        # self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

        # self.actions = actions.clone().to(self._device)
        # self.targets[:, self._shadow_hands.actuated_dof_indices] = self.actions
        # self._shadow_hands.set_joint_position_targets(self.targets)
        self._shadow_hands.set_velocities(torch.tensor([10.,0.,0., 0.,0.,0.]))
        # self._shadow_hands.set_world_poses(positions = torch.tensor([10., 0., 0.5], device=self._device).unsqueeze(0))
        pass

    def get_observations(self) -> dict:
        pass


    # def reset_idx(self, env_ids):
    #     indices = env_ids.to(dtype=torch.int32)
    #     num_indices = len(indices)

    #     # Reset Franka robots
    #     pos = tensor_clamp(
    #         self.franka_default_dof_pos.unsqueeze(0)
    #         + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
    #         self.franka_dof_lower_limits,
    #         self.franka_dof_upper_limits,
    #     )
    #     dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
    #     dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
    #     dof_pos[:, :] = pos
    #     self.franka_dof_targets[env_ids, :] = pos
    #     self.franka_dof_pos[env_ids, :] = pos

        
    #     self._frankas.set_joint_positions(dof_pos, indices=indices)
    #     self._frankas.set_joint_velocities(dof_vel, indices=indices)
    #     self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)

    #     # Reset target positions
    #     pos = tensor_clamp(
    #         self._target_position.unsqueeze(0)
    #         + 0.25 * (torch.rand((len(env_ids), 3), device=self._device) - 0.5),
    #         self._target_lower_bound,
    #         self._target_upper_bound,
    #     )
    #     dof_pos = torch.zeros((num_indices, 3), device=self._device)
    #     dof_pos[:, :] = pos + self._env_pos[env_ids]
    #     self._targets.set_world_poses(positions=dof_pos, indices=indices)




    #     # bookkeeping
    #     self.reset_buf[env_ids] = 0
    #     self.progress_buf[env_ids] = 0


    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        # Distance to target
        # d = torch.norm(self.hand_pos - self.target_pos, p=2, dim=1)
        # reward = torch.log(1 / (1.0 + d ** 2))

        # Difference in orientation
        # d = quat_diff_rad(self.hand_rot, self.target_rot)
        # reward += (1.0 / (torch.abs(d**2) + 0.1)) * 1.

        # Extra reward if it is close enough
        # reward = torch.where(torch.norm(self.hand_pos - self.target_pos, p=2, dim=1) < 0.05, reward + 1, reward)

        # self.rew_buf[:] = reward
        pass

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # reset if max episode length is exceeded
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

