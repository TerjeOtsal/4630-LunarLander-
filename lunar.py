import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        return self.layer_3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(1e6)
        self.batch_size = 100
        self.discount = 0.99
        self.tau = 0.005
    
    # Additional methods for training and updating networks will be needed here
    def select_action(self, state):
     if not isinstance(state, np.ndarray):
        state = np.array(state)
     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
     action = self.actor(state).cpu().data.numpy().flatten()
     return action

    def train(self):
     if len(self.replay_buffer) < self.batch_size:
        return
    
     state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
     state = torch.FloatTensor(state).to(device)
     action = torch.FloatTensor(action).to(device)
     reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
     next_state = torch.FloatTensor(next_state).to(device)
     done = torch.FloatTensor(done).to(device).unsqueeze(1)

    # Compute the target Q value
     with torch.no_grad():
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.discount * target_Q

    # Get current Q estimate
     current_Q = self.critic(state, action)

    # Compute critic loss
     critic_loss = nn.MSELoss()(current_Q, target_Q)

     # Optimize the critic
     self.critic_optimizer.zero_grad()
     critic_loss.backward()
     self.critic_optimizer.step()

     # Compute actor loss
     actor_loss = -self.critic(state, self.actor(state)).mean()

     # Optimize the actor
     self.actor_optimizer.zero_grad()
     actor_loss.backward()
     self.actor_optimizer.step()

     # Update the frozen target models
     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

     for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
         target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        

# Main Training Loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    # Note: You'll need to implement the training loop and methods for updating the networks.

    episodes = 10  # Number of episodes to train for
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        agent.train()  # Train the agent
        
        if done:
            print(f"Episode: {episode + 1}, Total Reward: {episode_reward}")
            break
