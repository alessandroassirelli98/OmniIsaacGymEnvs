import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.style.use("seaborn")

def parse_json_demo():
    df = []
    for root, dir, filenames in os.walk(f'{os.getcwd()}{"/demonstrations/data"}'):
        for filename in filenames:
            with open(f'{root}{"/"}{filename}') as f:
                data = json.load(f)
                df_tmp = pd.json_normalize(data["Isaac Sim Data"])
                df_tmp.columns = ["time", "timestep", "states", "actions", "rewards", "terminated", "applied_joint_act"]
                df.append(df_tmp)
    if len(df) == 1:
        df = df[0]
    else:
        df = pd.concat(df).reset_index()

    states = df["states"]
    actions = df["actions"]
    rewards = df["rewards"]
    terminated = df["terminated"]


    episode = []
    dim = len(states)
    for i, (s, a, r, t) in enumerate(zip(states, actions, rewards, terminated)):
        if i == dim-1: break
        timestep = {}
        timestep["states"] = s
        timestep["actions"] = a
        timestep["rewards"] = r
        timestep["terminated"] = t
        # TODO fix unavailable next states
        if not bool(t[0]):
            timestep["next_states"] = states[i + 1]
        else:
            timestep["next_states"] = s
            # a = np.empty_like(s)[:]
            # a[:] = np.nan
            # timestep["next_states"] = a

        episode.append(timestep)

    return episode

if __name__ == "__main__":
    e = parse_json_demo()
    expert_len = len(e)
    which_episode = 2

    ep_ret = []
    r = []
    actions = []
    done = False
    episode_cnt = 0
    t_ = 0
    for t in range(expert_len):
        done = bool(e[t]["terminated"][0])
        if not done:
            r.append(pow(0.99, t_) * e[t]["rewards"][0])
            actions.append(np.array(e[t]["actions"]))
            t_ += 1

        else:
            ep_ret.append(r)
            r = []
            done = False
            episode_cnt += 1
            t_ = 0
    print("Average Return: ", np.mean(ep_ret))
    print("Demonstrations length: ", len(e))
    print("Number of Episodes: ", episode_cnt)
    plt.plot(np.array(r))
    plt.show()

    # for i in range(7):
    #     plt.subplot(2,4, i+1)
    #     plt.title(f'action {i}')
    #     plt.plot(actions[i, :])
    # plt.show()