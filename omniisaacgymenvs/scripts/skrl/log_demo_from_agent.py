import torch
import torch.nn as nn
import sys
import git
import os


from omniisaacgymenvs.utils.logger import Logger
# import the skrl components to build the RL system
from skrl.agents.torch.ppofd import PPOFD, PPOFD_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed


set_seed(42)  # e.g. `set_seed(42)` for fixed seed
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}
    
    def reset_std(self):
        with torch.no_grad():
            self.log_std_parameter.zero_()
    
    def make_deterministic(self):
        with torch.no_grad():
            self.log_std_parameter.fill_(torch.finfo(torch.float32).min)
    
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction) # here we clip inside
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
        )

        self.mean_layer = nn.Sequential(nn.Linear(256, self.num_actions), nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(256, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}
        
    def make_deterministic(self):
        with torch.no_grad():
            self.log_std_parameter.fill_(torch.finfo(torch.float32).min)


# load and wrap the Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="FrankaCabinet", headless=False, num_envs=1)
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
samppling_demo_memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device, False)
models["value"] = Critic(env.observation_space, env.action_space, device, False)
# models["policy"] = Shared(env.observation_space, env.action_space, device)
# models["value"] = models["policy"]

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPOFD_DEFAULT_CONFIG.copy()

cfg["pretrain"] = False
cfg["pretrainer_epochs"] = 15
cfg["pretrainer_lr"] = 1e-3
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4  # 16 * 8192 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.001
cfg["value_loss_scale"] = 2.0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["kl_threshold"] = 0.008

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

checkpoint_path = "/home/alessandro.assirelli/devel/git-repos/OmniIsaacGymEnvs/omniisaacgymenvs/runs/torch/DianaTekken/24-07-14_14-29-14-845169_PPOFD/checkpoints/best_agent.pt"
agent.load(checkpoint_path)
agent.set_running_mode("eval")

states, infos = env.reset()
    # Saving path
for i in range(15):
    dire = f'{os.getcwd()}{"/demonstrations/data"}'
    file_path = f'{dire}{"/"}{str(i)}{".json"}'
    if not os.path.isfile(file_path):
        print("Saving demonstration in: " + file_path)
        break
    if i == 14:
        print("Directory busy")
        exit()

log = Logger(env)
log.start_logging(file_path)             

t = 0
while t < 24000:
    # compute actions
    with torch.no_grad():
        actions = agent.act(states, timestep=0, timesteps=0)[0]
        # step the environments
        next_states, rewards, terminated, truncated, infos = env.step(actions)

        log.logging_step()
    # reset environments
    if env.num_envs > 1:
        states = next_states
    else:
        if terminated.any() or truncated.any():
            with torch.no_grad():
                states, infos = env.reset()
        else:
            states = next_states
    t += 1
    print(t)

    #if kill:  # if key 'q' is pressed 
log.save_log()
env._simulation_app.close()
print(f'Saving demo in {file_path}')

            

