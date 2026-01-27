import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", render_mode="human")

print(env.observation_space)
print(env.action_space)
print(env.unwrapped.P[0][2])

env.reset()
# env.step(1)
env.render()

next_state, reward, done,trans_prob, info = env.step(1) 
print(next_state, reward, done, trans_prob, info)

# RANDOM POLICY
# random_action = env.action_space.sample()
# next_state, reward, done,trans_prob, info = env.step(random_action) 

# for multiple episodes
episode = 10
for m in range(episode): 
    rewards = []
    env.reset()
    num_timesteps = 10
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        next_state, reward, done, trans_prob, info = env.step(random_action)
        rewards.append(reward)
        env.render()
        if done: 
            break
    print(f"Return for Episode {m+1}: {sum(rewards)}")


