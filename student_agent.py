import numpy as np
import pickle
import random
import gym

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def get_action(obs):
    # Convert obs (tuple) to integer if required
    if isinstance(obs, int):
        state = obs
    else:
        # 如果 obs 不是整數，轉換成正確的索引
        taxi_row, taxi_col, pass_idx, dest_idx = obs
        state = env.unwrapped.encode(taxi_row, taxi_col, pass_idx, dest_idx)

    # 如果狀態超出 Q-table 範圍，選擇隨機行為 (保險)
    if state >= len(Q_table):
        return np.random.choice([0, 1, 2, 3, 4, 5])

    # 如果 Q-table 中有資料，選擇最佳行為
    return np.argmax(Q_table[state])
