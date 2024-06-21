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
                df_tmp.columns = ["time", "timestep", "states", "actions", "ee_pos", "ee_pos_des", "rewards", "terminated", "applied_joint_act"]
                df.append(df_tmp)
    if len(df) == 1:
        df = df[0]
    else:
        df = pd.concat(df).reset_index()

    actions = df["ee_pos_des"]
    ee_vel = df["ee_pos"]


    episode = []
    dim = len(actions)
    for i, (vdes, v) in enumerate(zip(actions,ee_vel)):
        if i == dim-1: break
        timestep = {}
        timestep["d_pos"] = vdes
        timestep["pos"] = v
        episode.append(timestep)

    return episode

if __name__ == "__main__":
    e = parse_json_demo()
    expert_len = len(e)
    which_episode = 0

    vdes = np.zeros((3, expert_len))
    v = np.zeros((3, expert_len))

    for t in range(expert_len):
        vdes[:, t] = e[t]["d_pos"][0][:3]
        v[:, t] = e[t]["pos"][0][:3]

    for i in range(3):
        plt.subplot(1,3, i+1)
        plt.plot(vdes[i, :])
        plt.plot(v[i, :])
        plt.legend(["desired", "real"])
    plt.show()
        