import gym
import numpy as np

# Environment and Hyperparameters
env = gym.make("LunarLanderContinuous-v2", render_mode="human")  # Choose rendering mode
episodes = 10  # Number of training episodes

for episode in range(episodes):
    observation = env.reset()  # Reset environment for each episode
    total_reward = 0
    done = False

    while not done:
        
        action = np.random.uniform(env.action_space.low, env.action_space.high, size=env.action_space.shape)

        next_state, reward, done, info, *_ = env.step(action)  # Take a step

        total_reward += reward

    print(f"Episode {episode+1} reward: {total_reward}")  # Print episode reward

env.close()  # Close the environment after training
