import math

import numpy as np
import torch
from gym import spaces
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import get_euler_xyz, quat_diff_rad
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, VisualSphere, FixedSphere
from omniisaacgymenvs.robots.articulations.tekken import Tekken
from omniisaacgymenvs.robots.articulations.views.tekken_view import TekkenView



class TekkenTask(RLTask):

    def __init__(self, name: str, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.robots_to_log = []

        self.hithand_cad_translation = torch.tensor([0.0, -0.15, 0.])
        self.hithand_old_translation = torch.tensor([0.0, 0.15, 0.])

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 14 + 3 + 3
        self._num_actions = 15
        self.action_space = spaces.Box(np.zeros(self._num_actions), np.ones(self._num_actions) * 1.0)

        RLTask.__init__(self, name, env)


    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_tekken(name="hithand_cad",
                        usd_path="C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/models/tekken_cad/tekken_cad.usd",
                        translation=self.hithand_cad_translation)

        super().set_up_scene(scene)

        self._tekkens = TekkenView(prim_paths_expr="/World/envs/.*/hithand_cad", name="tekken_view")
        self.robots_to_log.append(self._tekkens) # Robot that gets logged by the logger
        scene.add(self._tekkens)  # add view to scene for initialization

        
    def get_tekken(self, name, usd_path, translation):
        hithand = Tekken(prim_path=self.default_zero_env_path + '/' + name,
                              usd_path=usd_path,
                              name=name,
                              translation=translation)
        self._sim_config.apply_articulation_settings(name, get_prim_at_path(hithand.prim_path), self._sim_config.parse_actor_config(name))
        
        
    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.num_tekken_dofs = self._tekkens.num_dof
        self.actuated_dof_indices = self._tekkens.actuated_dof_indices
        self.tekken_dof_targets = torch.zeros((self.num_envs, self.num_tekken_dofs), device=self._device)
        dof_limits = self._tekkens.get_dof_limits()
        self.tekken_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.tekken_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self._tekkens.set_joint_positions(self.tekken_dof_targets)
        self._tekkens.set_joint_velocities(torch.zeros((self.num_envs, self.num_tekken_dofs), device=self._device))
        self._tekkens.set_joint_position_targets(self.tekken_dof_targets)
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

        self.actions = actions.clone().to(self._device)
        targets = self.tekken_dof_targets[:, self.actuated_dof_indices] + self.dt * self.actions * self.action_scale
        self.tekken_dof_targets[:, self.actuated_dof_indices] = targets
        env_ids_int32 = torch.arange(self._tekkens.count, dtype=torch.int32, device=self._device)
        self._tekkens.set_joint_position_targets(self.tekken_dof_targets, indices=env_ids_int32)


        

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

