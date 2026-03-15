import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode = "human")
observation, info = env.reset()
total_reward = 0
for _ in range(100) :
    action = env.action_space.sample()
    observation, reward, termination, truncation, info = env.step(action)
    total_reward += reward
    if termination or truncation:
        observation, info = env.reset()

env.close()

