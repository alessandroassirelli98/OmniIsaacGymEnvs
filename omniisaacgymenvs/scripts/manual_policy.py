from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omniisaacgymenvs.tasks.diana_tekken_manual_control import DianaTekkenManualControlTask
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omniisaacgymenvs.utils.input_manager import KeyboardManager

import carb

my_world = World(stage_units_in_meters=1.0)
save_path = "C:/Users/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/demonstrations/1.json"
data_logger = my_world.get_data_logger() # a DataLogger object is defined in the World by default

replay_scene = True

my_task = DianaTekkenManualControlTask(name="diana_tekken_manual")
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("diana_tekken_manual").get_params()
robot_name = task_params["robot_name"]["value"]
diana_tekken = my_world.scene.get_object(robot_name)
articulation_controller = diana_tekken.get_articulation_controller()

action = np.zeros(4)
close_hand = np.zeros(1, dtype=np.int16)         

input_manager = KeyboardManager(action, close_hand)

def start_logging():
    task = my_task

    # A data logging function is called at every time step index if the data logger is started already.
    # We define the function here. The tasks and scene are passed to this function when called.
    def frame_logging_func(tasks, scene):
        # return always a dict
        return {task.name:  {"obs_buf" : task.obs_buf.tolist(),
                             "action" : task._robot.get_applied_action().joint_positions.tolist()} }
        
    
    data_logger.add_data_frame_logging_func(frame_logging_func)
    data_logger.start()

def save_log(save_path):
    data_logger.save(save_path)

if replay_scene:
    data_logger.load(log_path = save_path)

is_init = False
while simulation_app.is_running():          
    my_world.step(render=True)
    if my_world.is_playing():
        if not replay_scene:
            if not is_init:
                start_logging()
                is_init = True
            if input_manager.kill: 
                save_log(save_path)
                simulation_app.close()
            if my_world.current_time_step_index == 0:
                my_world.reset()
            
            my_task.update(action, close_hand)
            observations = my_world.get_observations()
        else:
            if my_world.current_time_step_index < data_logger.get_num_of_data_frames():
                data_frame = data_logger.get_data_frame(data_frame_index = my_world.current_time_step_index)
                articulation_controller.apply_action(ArticulationAction(joint_positions = data_frame.data['diana_tekken_manual']['action']))

simulation_app.close()

 

 