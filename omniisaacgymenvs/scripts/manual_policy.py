from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import keyboard
from omniisaacgymenvs.tasks.diana_tekken_manual_control import DianaTekkenManualControlTask
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core import World
from omni.isaac.franka import KinematicsSolver
import carb

my_world = World(stage_units_in_meters=1.0)
my_task = DianaTekkenManualControlTask(name="diana_tekken_manual")
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("diana_tekken_manual").get_params()
robot_name = task_params["robot_name"]["value"]
diana_tekken = my_world.scene.get_object(robot_name)
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        observations = my_world.get_observations()
        action = np.zeros(4)
        if (keyboard.is_pressed('up')):
            action[0] = 1
        if (keyboard.is_pressed('down')):
            action[0] = -1
        if (keyboard.is_pressed('left')):
            action[1] = 1
        if (keyboard.is_pressed('right')):
            action[1] = -1
        if (keyboard.is_pressed('w')):
            action[2] = 1
        if (keyboard.is_pressed('s')):
            action[2] = -1
        if (keyboard.is_pressed('a')):
            action[3] = 1
        if (keyboard.is_pressed('z')):
            action[3] = -1
        my_task.update(action)

simulation_app.close()

