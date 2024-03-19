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
from omni.isaac.core.objects import FixedCuboid, DynamicSphere, DynamicCuboid, VisualCuboid, FixedSphere
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

        self._num_observations = 68
        self._num_actions = 3
        self.obs_buf = np.zeros(self._num_observations)

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
        self.get_pick_up_cube()
        self.get_reference_cube()

        super().set_up_scene(scene)
        self._palm_center = RigidPrimView(prim_paths_expr="/World/diana_tekken/palm_link_hithand", name="palm_centers_view", reset_xform_properties=False)

        scene.add(self._robot)
        scene.add(self._cube)
        scene.add(self._pick_up_cube)
        scene.add(self._ref_cube)
        scene.add(self._palm_center)



        self._ik = KinematicsSolver(self._robot)
        self._articulation_controller = self._robot.get_articulation_controller()
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
    
    def get_pick_up_cube(self):
        self._pick_up_cube_color = np.array([0.1, 0.9, 0.1], dtype=np.float32)
        self._pick_up_cube_position = np.array([0.5, 0., 0.48], dtype=np.float32)
        self._pick_up_cube_lower_bound = np.array([0.2, -0.5, 0.48], dtype=np.float32)
        self._pick_up_cube_upper_bound = np.array([0.8, 0.5, 0.48], dtype=np.float32)

        self._pick_up_cube = DynamicCuboid(prim_path= self.default_prim_path + "pick_up_cube",
                                  name="pick_up_cube",
                                  translation= self._pick_up_cube_position,
                                  scale=np.array([0.06, 0.06, 0.06]),
                                  color=self._pick_up_cube_color,
                                  mass = 0.03)

    def get_reference_cube(self):
        self._ref_cube = VisualCuboid(prim_path= self.default_prim_path  + "ref_cube",
                                  name="ref_cube",
                                  translation= np.array([0.4, 0., 0.8], dtype=np.float32),
                                  orientation=euler_angles_to_quats(np.array([0., -np.pi/6, 0.])),
                                  scale = np.array([0.02, 0.02, 0.02], dtype=np.float32),
                                  color=np.array([1, 0, 0], dtype=np.float32))
        
    def post_reset(self):
        self._robot.initialize_dof_indices()


    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        # implement logic to be performed before physics steps
        pass
    
    def update(self, delta, close_hand=False):
        # Move target position and orientation
        target_pos, target_rot = self._ref_cube.get_local_pose()
        rpy_target = quats_to_euler_angles(target_rot)
        target_pos += delta[:3] * 0.001
        rpy_target[1] += delta[3] * 0.001
        target_rot = euler_angles_to_quats(rpy_target)

        actions, succ = self._ik.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_rot)
        
        if close_hand[0] == 1:
            actions.joint_positions[self._robot.actuated_finger_dof_indices] = np.ones(len(self._robot.actuated_finger_dof_indices)) * np.pi/3
        else:
            actions.joint_positions[self._robot.actuated_finger_dof_indices] = np.zeros(len(self._robot.actuated_finger_dof_indices))

        if succ:
            self._articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")
        
        self._ref_cube.set_world_pose(position=target_pos, orientation=target_rot)



    def get_observations(self) -> dict:
        dof_pos = self._robot.get_joint_positions()
        dof_vel = self._robot.get_joint_velocities()
        self.hand_pos,  self.hand_rot = self._palm_center.get_local_poses()
        self.pick_up_cube_pos, self.pick_up_cube_rot = self._pick_up_cube.get_local_pose()

        self.obs_buf[:27] = dof_pos
        self.obs_buf[27:30] = self.hand_pos
        self.obs_buf[30:34] = self.hand_rot
        self.obs_buf[34:37] = self.pick_up_cube_pos
        self.obs_buf[37:41] = self.pick_up_cube_rot
        self.obs_buf[41:68] = dof_vel

        return {self._robot.name: {"obs_buf": self.obs_buf}}


    def calculate_metrics(self) -> None:
        pass

    def is_done(self) -> None:
        pass

