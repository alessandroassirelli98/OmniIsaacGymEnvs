import torch
import torch.nn as nn
import sys
import git

# import the skrl components to build the RL system
from skrl.agents.torch.ppofd import PPOFD, PPOFD_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer, Pretrainer
from skrl.utils import set_seed
from omniisaacgymenvs.demonstrations.demo_parser import parse_json_demo
from omniisaacgymenvs.utils.parse_algo_config import parse_arguments


# Check git commit
repo = git.Repo(search_parent_directories=True)
commit_hash = repo.head.object.hexsha

if repo.is_dirty():
    print("There are unstaged changes, please commit before run\n")
    exit()

else:
    print("Repo is clean, proceeeding to run \n")

ignore_args = ["headless", "task", "num_envs"] # These shouldn't be handled by this fcn
algo_config = parse_arguments(ignore_args)

if "random_seed" in algo_config.keys():
    random_seed = int(algo_config['random_seed'])
else:
    random_seed = 42
# seed for reproducibility
set_seed(random_seed)  # e.g. `set_seed(42)` for fixed seed

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())
                                 
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
    
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

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU()
                                 )
        self.value_layer = nn.Linear(64, 1)

    def compute(self, inputs, role):
        return self.value_layer(self.net(inputs["states"])), {}   

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction) # here we clip inside
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Sequential(nn.Linear(64, self.num_actions), nn.Tanh())
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
        
    def make_deterministic(self):
        with torch.no_grad():
            self.log_std_parameter.fill_(torch.finfo(torch.float32).min)


# load and wrap the Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="DianaTekken")
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
samppling_demo_memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
# models["policy"] = StochasticActor(env.observation_space, env.action_space, device, True)
# models["value"] = Critic(env.observation_space, env.action_space, device, False)
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
plot=True
cfg = PPOFD_DEFAULT_CONFIG.copy()
cfg["commit_hash"] = commit_hash

cfg["nn_type"] = "shared"

cfg["pretrain"] = False
cfg["pretrainer_epochs"] = 50
cfg["pretrainer_lr"] = 1e-3
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4  # 16 * 8192 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0001
cfg["value_loss_scale"] = 2.0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["kl_threshold"] = 0.008

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 800
cfg["experiment"]["directory"] = "runs/torch/DianaTekken"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {"tags" : ["PPOFD "],
                                     "project": "pick up trial 12 DOF finger_joints"}

for key, value in algo_config.items():
    print(key, value)
    if key == "checkpoint":
        pass
    elif key == "reward_shaper" and value == True:
        value = lambda rewards, timestep, timesteps: rewards * 0.01
    elif key == "reward_shaper" and value == False:
        value = None
    elif value == "SharedNetworks":
        models = {}
        models["policy"] = Shared(env.observation_space, env.action_space, device)
        models["value"] = models["policy"]
        cfg["nn_type"] = "shared"

    elif value == "SeparateNetworks":
        models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
        models["value"] = Critic(env.observation_space, env.action_space, device)
        cfg["nn_type"] = "separate"

    elif value == "RunningStandardScaler":
        value = RunningStandardScaler
    elif value == "KLAdaptiveRL":
        value = KLAdaptiveRL
    elif value == 'True':
        value = True
    elif value == 'False':
        value = False
    elif '.' in value:
        value = float(value)
    else:
        value = int(value)
    cfg[str(key)] = value
    print(f"Setting {key} to {value} of type {type(value)}")


# Buffer prefill
episode = parse_json_demo()
demo_size = len(episode)
demonstration_memory = RandomMemory(memory_size=demo_size, num_envs=1, device=device)

agent = PPOFD(models=models,
            memory=memory,
            demonstration_memory=demonstration_memory,
            sampling_demo_memory=samppling_demo_memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# demonstrations injection
if cfg["pretrain"]:
    transitions = []
    for tstep in episode:
        states = torch.tensor(tstep["states"], device=device)
        actions = torch.tensor(tstep["actions"], device=device)
        rewards = torch.tensor(tstep["rewards"], device=device)
        terminated = torch.tensor(tstep["terminated"], device=device)
        next_states = torch.tensor(tstep["next_states"], device=device)
        dict = {}
        dict["states"] = states
        dict["actions"] = actions
        dict["reward"] = rewards
        dict["next_states"] = next_states
        dict["terminated"] = terminated
        transitions.append(dict)
        demonstration_memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,terminated=terminated)

    # trainer.pre_train(transitions, 10)
    pt = Pretrainer(agent=agent,
                    transitions=transitions,
                    lr=cfg["pretrainer_lr"],
                epochs=cfg["pretrainer_epochs"],
                batch_size=32)

# start training
if cfg["checkpoint"]:
    agent.load(cfg["checkpoint"])

if cfg["pretrain"] and not cfg["test"]:
    import matplotlib.pyplot as plt

    replay_actions = pt.test_bc()
    pt.train_bc()
    
    if plot:
        test_cpu = pt.test_policy_loss.cpu()
        plt.title("timestep error")
        plt.plot(test_cpu)
        plt.ylabel("error")
        plt.xlabel("timestep")
        plt.show()

        plt.title("BC loss")
        plt.plot(pt.log_policy_loss.cpu())
        plt.show()

        replay_actions = pt.test_bc()
        test_cpu = pt.test_policy_loss.cpu()
        plt.title("Acion test loss")
        plt.plot(test_cpu)
        plt.ylabel("mse")
        plt.xlabel("timestep")
        plt.show()

        std_cpu = pt.log_std.cpu()
        plt.title("Actions std")
        for i in range(std_cpu.shape[1]):
            plt.plot(std_cpu[:,i])
        plt.ylabel("std")
        plt.xlabel("Epoch")
        plt.show()

        mean_cpu = pt.log_mse.cpu()
        plt.title("Actions mse")
        for i in range(mean_cpu.shape[1]):
            plt.plot(mean_cpu[:,i])
        plt.ylabel("mse")
        plt.xlabel("Epoch")
        plt.show()

if not cfg["test"]:
    trainer.train()
else:
    # agent.policy.make_deterministic()
    trainer.eval()
