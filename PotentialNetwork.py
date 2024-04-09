import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Neural Network for Policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))  # Log standard deviation

    def forward(self, x):
        mu = self.network(x)
        sigma = torch.exp(self.sigma).expand_as(mu)
        return Normal(mu, sigma)

def train_policy_gradient():
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    episodes = 10

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        done = False
        total_reward = 0

        while not done:
            dist = policy(state)
            action = dist.sample()
            action = action.clamp(env.action_space.low[0], env.action_space.high[0])
            next_state, reward, done, info, *_ = env.step(action.numpy())
            total_reward += reward

            # Training
            optimizer.zero_grad()
            loss = -dist.log_prob(action) * reward  # Policy gradient update
            loss.mean().backward()
            optimizer.step()

            state = torch.FloatTensor(next_state)

        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    env.close()

train_policy_gradient()