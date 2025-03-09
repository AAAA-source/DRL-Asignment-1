# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym


# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def get_action(obs):
    # Convert obs (tuple) to integer if required
    state = obs if isinstance(obs, int) else hash(obs) % len(Q_table)
    
    # Ensure the observation exists in Q-table, otherwise pick a random action
    if state in Q_table:
        return np.argmax(Q_table[state])  # Select best action from Q-table
    else:
        return np.random.choice([0, 1, 2, 3, 4, 5])  # Random fallback
