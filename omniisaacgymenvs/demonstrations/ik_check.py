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
                df_tmp.columns = ["time", "timestep", "states", "actions", "ee_vel", "rewards", "terminated", "applied_joint_act"]
                df.append(df_tmp)
    if len(df) == 1:
        df = df[0]
    else:
        df = pd.concat(df).reset_index()

    actions = df["actions"]
    ee_vel = df["ee_vel"]


    episode = []
    dim = len(actions)
    for i, (vdes, v) in enumerate(zip(actions,ee_vel)):
        if i == dim-1: break
        timestep = {}
        timestep["vdes"] = vdes
        timestep["v"] = v
        episode.append(timestep)

    return episode

if __name__ == "__main__":
    e = parse_json_demo()
    expert_len = len(e)
    which_episode = 0

    vdes = np.zeros((6, expert_len))
    v = np.zeros((6, expert_len))
    print(e[0]["v"][0])

    for t in range(expert_len):
        vdes[:, t] = e[t]["vdes"][0][:6]
        v[:, t] = e[t]["v"][0][:6]

    for i in range(6):
        plt.subplot(2,3, i+1)
        plt.plot(vdes[i, :])
        plt.plot(v[i, :])
        plt.legend(["desired", "real"])
    plt.show()
        