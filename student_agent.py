import numpy as np
import pickle
import random
from collections import defaultdict, deque
from form_state import State  

# 全局變數
q_table = None
state_tracker = None
action = None
stations = None
visited_states = deque(maxlen=5)  
temperature = 1.0  
P_threshold = 0.99
print("Student agent loaded")

def load_table(filename="Q3M.pkl"):
    global q_table
    if q_table is None:
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
        q_table = defaultdict(lambda: np.zeros(6), q_table) 

    return q_table

def detect_loop(state):
    return visited_states.count(state) >= 3  

def softmax(x, temp=1.0):
    x = np.array(x) / temp 
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def top_p_sampling(probs, P_threshold=0.9):
    sorted_indices = np.argsort(probs)[::-1]  
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs) 

    selected_indices = sorted_indices[cumulative_probs <= P_threshold]
    if len(selected_indices) == 0:
        selected_indices = sorted_indices[:1] 

    return np.random.choice(selected_indices)  

def get_action(obs):
    global q_table, state_tracker, action, stations, visited_states, temperature

    if q_table is None:
        q_table = load_table()

    if state_tracker is None:
        state_tracker = State(obs)  
        state = state_tracker._compute_full_state()
    elif stations != obs[2:10]:
        state_tracker = State(obs)
        state = state_tracker._compute_full_state()
        temperature = 0.25
    else:
        state_tracker.update(obs, action)
        state = state_tracker._compute_full_state()

    state = tuple(state_tracker.full_state)

    if state not in q_table:
        print(state)
        q_table[state] = np.zeros(6)

    q_values = q_table[state]

    visited_states.append(state)
    if detect_loop(state):  
        temperature = min(temperature * 2, 4.0)  
    else:
        temperature = max(temperature * 0.8, 0.25) 


    probs = softmax(q_values, temperature)
    action = top_p_sampling(probs, P_threshold)
    state_tracker.pre_action = action
    stations = obs[2:10]

    return action
