import gymnasium as gym
from collections import defaultdict
import pandas as pd

# Title - Implementation of Single-Visit Monte Carlo prediction for blackjack environment

env = gym.make('Blackjack-v1', render_mode="human")

def policy(state):
    return 0 if state[0] > 19 else 1

num_timesteps = 100

def generate_episode(policy):
    episode = []
    state, _ = env.reset()

    for t in range(num_timesteps):
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        if terminated or truncated:
            break

        state = next_state

    return episode

# dictionaries to store returns and visit counts
total_return = defaultdict(float)
N = defaultdict(int)

num_iterations = 10

# Monte Carlo Policy Evaluation
for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)

    for t, state in enumerate(states):
        # compute return from time t onward
        if state not in states[0:t]:
            R = sum(rewards[t:])
            total_return[state] += R
            N[state] += 1

# Convert to DataFrames
total_return_df = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
N_df = pd.DataFrame(N.items(), columns=['state', 'N'])

# Merge both df using states column
df = pd.merge(total_return_df, N_df, on='state')

# Compute value function
df['value'] = df['total_return'] / df['N']
print(df)

'''  
The algorithm successfully estimated state values by averaging returns across multiple episodes, producing a DataFrame of states with their visit counts and corresponding value estimates.
'''