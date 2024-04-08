import torch
import torch.nn as nn
import sys

# import the skrl components to build the RL system
from skrl.agents.torch.ppofd import PPOFD, PPOFD_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omniisaacgymenvs.demonstrations.demo_parser import parse_json_demo


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 )

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

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


# load and wrap the Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="DianaTekken")
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPOFD_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 16 * 8192 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 4000
cfg["experiment"]["directory"] = "runs/torch/DianaTekken"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {"tags" : ["PPOFD"], 
                                     "project": "DrillPickUpAlgoTrials"}

defined = False
for arg in sys.argv:
    if arg.startswith("test="):
        defined = True
        break
# get wandb usage from command line arguments
if defined:
    test = bool(arg.split("test=")[1].split(" ")[0])
    if test: cfg["experiment"]["wandb"] = False
else: 
    test = False

defined = False
for arg in sys.argv:
    if arg.startswith("checkpoint="):
        defined = True
        break
# get wandb usage from command line arguments
if defined:
    checkpoint_path = (arg.split("checkpoint=")[1].split(" ")[0])
else: 
    checkpoint_path = None
        

# Buffer prefill
episode = parse_json_demo()
demo_size = len(episode)
demonstration_memory = RandomMemory(memory_size=demo_size, num_envs=1, device=device)

agent = PPOFD(models=models,
            memory=memory,
            demonstration_memory=demonstration_memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# demonstrations injection
for tstep in episode:
    states = torch.tensor(tstep["states"], device=device)
    actions = torch.tensor(tstep["actions"], device=device)
    rewards = torch.tensor(tstep["rewards"], device=device)
    terminated = torch.tensor(tstep["terminated"], device=device)
    next_states = torch.tensor(tstep["next_states"], device=device)
    demonstration_memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,terminated=terminated)

# start training
if checkpoint_path:
    agent.load(checkpoint_path)

if not test:
    trainer.train()
else:
    trainer.eval()
