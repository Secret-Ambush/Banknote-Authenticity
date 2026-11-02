# Write a python program to design a Hopfield Network which stores 4 vectors 

import numpy as np

def train_hopfield(patterns, size):
    weights = np.zeros((size, size))
    for p in patterns:
        p = np.array(p).reshape(size, 1)
        weights += np.dot(p, p.T)
    np.fill_diagonal(weights, 0)
    return weights
  
def asynchronous_recall(weights, noisy_input, max_iter=100):
    state = np.array(noisy_input)
    size = len(state)
    for _ in range(max_iter):
        prev_state = state.copy()
        for i in range(size):
            net_input = np.dot(weights[i], state)
            state[i] = 1 if net_input > 0 else -1 if net_input < 0 else state[i]
        if np.array_equal(state, prev_state):
            break
    return state
  
patterns = [
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
]

size = 4
weights = train_hopfield(patterns, size)

noisy_input = [-1, -1, 1, -1]
recalled_state = asynchronous_recall(weights, noisy_input)

print("Noisy Input:", noisy_input)
print("Recalled Pattern:", recalled_state.tolist())