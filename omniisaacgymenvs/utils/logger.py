
class Logger():
    def __init__(self, env):
        self.env = env

    def start_logging(self, save_path):
        self.data_logger = self.env._world.get_data_logger() # a DataLogger object is defined in the World by default
        self._save_path = save_path
        robot = self.env._task._frankas
        task = self.env._task

        # A data logging function is called at every time step index if the data logger is started already.
        # We define the function here. The tasks and scene are passed to this function when called.
        def frame_logging_func(tasks, scene):
            # return always a dict
            
            return  {robot.name : {"states" : task.obs_buf.tolist(),
                                   "actions" : task.actions.tolist(),
                                    "rewards": task.rew_buf.tolist(),
                                    "terminated": task.reset_buf.tolist()}}
        
        self.data_logger.add_data_frame_logging_func(frame_logging_func)
        
        # self.data_logger.start() # Do Not execute this, otherwise the logger will be called inside the world step. Instead we want to call it after post step

    
    def logging_step(self):
        data = self.data_logger._data_frame_logging_func(tasks=self.env._world.get_current_tasks(), scene=self.env._world.scene)
        self.data_logger.add_data(
            data=data, current_time_step=self.env._world.current_time_step_index, current_time=self.env._world.current_time
        )

    def save_log(self):
        self.data_logger.save(self._save_path)