# PRACTICAL 6: Q-learning in a small grid environment
# Install: python -m pip install numpy

import numpy as np
import random

grid_size = 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 3), (3, 1)]

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000
max_steps = 100

# Actions: up, down, left, right
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
q_table = np.zeros((grid_size, grid_size, len(actions)))

def valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size and pos not in obstacles

def next_state(state, action):
    dx, dy = actions[action]
    candidate = (state[0] + dx, state[1] + dy)
    return candidate if valid(candidate) else state

def reward(state):
    if state == goal:
        return 10
    return -1

for episode in range(episodes):
    state = start
    total_reward = 0
    for step in range(max_steps):
        if random.random() < epsilon:
            action = random.choice(list(actions.keys()))
        else:
            action = int(np.argmax(q_table[state[0], state[1]]))

        new_state = next_state(state, action)
        r = reward(new_state)
        old = q_table[state[0], state[1], action]
        future = np.max(q_table[new_state[0], new_state[1]])
        q_table[state[0], state[1], action] = old + alpha * (
            r + gamma * future - old
        )
        state = new_state
        total_reward += r
        if state == goal:
            break
    if (episode + 1) % 100 == 0:
        print("Episode", episode + 1, "Total reward:", total_reward)

# Greedy test; a step limit prevents endless loops.
state = start
path = [state]
for _ in range(50):
    if state == goal:
        break
    action = int(np.argmax(q_table[state[0], state[1]]))
    state = next_state(state, action)
    path.append(state)

print("Learned path:", path)
if path[-1] == goal:
    print("Goal reached!")
else:
    print("Goal not reached in the step limit. Run again to train longer.")
