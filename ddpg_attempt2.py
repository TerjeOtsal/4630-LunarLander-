import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = deque(maxlen=int(max_size))

    def add(self, transition):
        self.storage.append(transition)

    def sample(self, batch_size=64):
        return random.sample(self.storage, batch_size)

class OUNoise(object):
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + dx
        return self.state * self.scale

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.replay_buffer = ReplayBuffer()
        self.ou_noise = OUNoise(action_dim)
        self.max_action = max_action
    
    def select_action(self, state):
     state = torch.FloatTensor(state).unsqueeze(0)  # Ensure it's a float tensor with a batch dimension
     action = self.actor(state).cpu().data.numpy().flatten()
     return action + self.ou_noise.noise()


    def train(self):
        if len(self.replay_buffer.storage) < batch_size:
            return

        sample = self.replay_buffer.sample(batch_size)
        state, next_state, action, reward, done = zip(*sample)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Initialize the environment and agent
env = gym.make("LunarLanderContinuous-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

gamma = 0.99
tau = 0.001
batch_size = 64

agent = DDPGAgent(state_dim, action_dim, max_action)

episodes = 1000  # Or the number of episodes you prefer
for episode in range(episodes):
    state = env.reset()[0]  # Adjust here to take the first element
    agent.ou_noise.reset()  # Reset the noise process for the new episode
    episode_reward = 0

    while True:
        action = agent.select_action(state)  # Now state is correctly formatted
        next_state, reward, done, _ = env.step(action)
        next_state = next_state[0]  # Adjust here as well for the next state
        episode_reward += reward

        agent.replay_buffer.add((state, next_state, action, reward, done))
        state = next_state

        agent.train()
        agent.update_target_networks()

        if done:
            print(f"Episode: {episode + 1}, Reward: {episode_reward}")
            break
