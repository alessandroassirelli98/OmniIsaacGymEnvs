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
    which_episode = 2
    ep_length = 800
    ep_ret = []
    r = []
    actions = np.zeros((7, ep_length))
    for t in range(ep_length):
        r.append(e[(ep_length * which_episode + t)]["rewards"])
        actions[:, t] = np.array(e[(ep_length * which_episode + t)]["actions"])
        ep_ret.append(r)
    print("Average Return: ", np.mean(ep_ret))
    print("DEmonstrations length: ", len(e))
    plt.plot(np.array(r))
    plt.show()

    for i in range(7):
        plt.subplot(2,4, i+1)
        plt.title(f'action {i}')
        plt.plot(actions[i, :])
    plt.show()