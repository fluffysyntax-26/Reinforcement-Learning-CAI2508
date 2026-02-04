import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# The folder 'video' will store .mp4 files
env = gym.make("CartPole-v1", render_mode='rgb_array')
env = RecordVideo(env, video_folder="./cartpole_videos", episode_trigger=lambda x: x % 10 == 0)

max_return = 0
best_ep = 0
num_episodes = 50

# print agent policy
print(f"Agent Policy: Pi(a|s) = 0.5 for action 0 (Left), 0.5 for action 1 (Right)")

for m in range(num_episodes): 
    state, info = env.reset()
    num_timesteps = 1000
    current_return = 0
    env.reset()
    
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        
        next_state, reward, done, info, trans_prob = env.step(random_action)
        
        current_return += reward
        env.render()
        
        if done: 
            break
            
    if (m + 1) % 10 == 0: 
        print(f"Episode {m + 1} | Return: {current_return}")
        
    if current_return > max_return: 
        max_return = current_return
        best_ep = m + 1


print(f"\nMax Return: {max_return} achieved at Episode: {best_ep}")

input("Press enter to close")
env.close()