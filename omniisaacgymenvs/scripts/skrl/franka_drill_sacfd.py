import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omniisaacgymenvs.demonstrations.demo_parser import parse_json_demo
from omniisaacgymenvs.utils.parse_algo_config import parse_arguments



# seed for reproducibility
ignore_args = ["headless", "task", "num_envs"] # These shouldn't be handled by this fcn
algo_config = parse_arguments(ignore_args)
if "random_seed" in algo_config.keys():
    rs = int(algo_config["random_seed"])
    print("set random seed ", rs)
    exit
else:
    rs = 42

# seed for reproducibility
set_seed(rs)  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="FrankaCabinet", num_envs=1024)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["random_seed"] = rs
cfg["pretrain"] = False
cfg["test"] = False


cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 200
cfg["experiment"]["directory"] = "runs/torch/DianaTekken"
cfg["experiment"]["wandb"] = False
cfg["experiment"]["wandb_kwargs"] = {"tags" : ["sacfd "],
                                     "project": "franka_tekken 12 dof js rev5"}

for key, value in algo_config.items():
    print(key, value)
    if key == "checkpoint":
        pass
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


agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Buffer prefill
episode = parse_json_demo()
demo_size = len(episode)

if cfg["pretrain"] and not cfg["test"]:
    transitions = []
    mem_idx = 0
    env_id = 0  
    for i, tstep in enumerate(episode):
        if i == (len(episode) // env.num_envs) * env.num_envs:
            break
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

        memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                        terminated=terminated, truncated=terminated)

        env_id += 1
        if env_id == env.num_envs:
            env_id = 0
            mem_idx += 1

# start training
trainer.train()

# path = "/home/ows-user/devel/git-repos/OmniIsaacGymEnvs_forked/omniisaacgymenvs/runs/torch/DianaTekken/24-06-13_19-16-21-970876_SAC/checkpoints/best_agent.pt"
# agent.load(path)
# trainer.eval()
