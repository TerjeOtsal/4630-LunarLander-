#needed Imports
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch as T
import random
import gym
import matplotlib.pyplot as plt
import os
# Ornstein-Uhlenbeck Action Noise
class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        # Initialize parameters
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # Generate noise
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        # Reset noise to initial state
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# Replay Buffer for Experience Replay
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        # Initialize memory buffer
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        # Store transition in memory
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Sample a batch from memory
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        # Initialize critic network layers
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Define fully connected layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Batch normalization layers for stabilizing training
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Action value layer to incorporate action information
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        # Output layer to calculate Q-values
        self.q = nn.Linear(self.fc2_dims, 1)

        # Adam optimizer for training
        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)

    def forward(self, state, action):
        # Forward pass of critic network
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # Incorporate action information
        action_value = self.action_value(action)

        # Combine state and action values
        state_action_value = F.relu(T.add(state_value, action_value))

        # Calculate Q-values
        state_action_value = self.q(state_action_value)

        return state_action_value


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        # Initialize actor network layers
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # Define fully connected layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Batch normalization layers for stabilizing training
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # Output layer for mean action values
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
    # Adam optimizer for training
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))   # Scale output to range [-1, 1] for continuous action space

        return x


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64):
           # Initialize DDPG agent
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
           # Choose action with noise
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
         # Remember experience
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
         # Update critic and actor networks
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
          # Update target networks
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

def save_models():
    if not os.path.exists('SavedModel'):
        os.makedirs('SavedModel')
    T.save(agent.actor.state_dict(), 'SavedModel/actor.pth')
    T.save(agent.critic.state_dict(), 'SavedModel/critic.pth')
    T.save(agent.target_actor.state_dict(), 'SavedModel/target_actor.pth')
    T.save(agent.target_critic.state_dict(), 'SavedModel/target_critic.pth')

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.axhline(y=200, color='r', linestyle='-', label='Successful Landing Threshold (200)') 
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def plot_best_scores(x, scores, figure_file):
    plt.figure()
    plt.plot(x, scores, label='Best Score per Episode')
    plt.title('Best Scores Across All Episodes')
    plt.axhline(y=200, color='r', linestyle='-', label='Successful Landing Threshold (200)')
    plt.legend()
    plt.savefig(figure_file)   
    
if __name__ == '__main__':
    # LunarLander environment
    env = gym.make('LunarLander-v2', continuous=True, gravity=random.uniform(-12, 0), turbulence_power=np.random.uniform(0.5, 2.0), enable_wind=True, wind_power=random.uniform(0, 20), render_mode="human")
    # Initialize the agent
    agent = Agent(alpha=0.0002, beta=0.002, 
                  input_dims=env.observation_space.shape, tau=0.005,
                  batch_size=96, fc1_dims=600, fc2_dims=450, 
                  n_actions=env.action_space.shape[0])
    n_games = 5000
    max_steps_per_episode = 1000  # Setting a limit on steps per episode
    score_history = []
    success_count = 0  # Counter for consecutive successes

    for i in range(n_games):
          # Reset environment and episode variables
        observation, _ = env.reset()
        done = False
        score = 0
        agent.noise.reset()
         # Episode loop
        for j in range(max_steps_per_episode):
                # Choose action, take step, and learn
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
             # Update episode score and observation
            score += reward
            observation = observation_
            if done:
                break
            # Append episode score and calculate average score
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print(f'episode {i} score {score:.1f} average score {avg_score:.1f}')

        if avg_score >= 200:
            success_count += 1
        else:
            success_count = 0  # Reset counter if the condition is not met

        if success_count >= 5:  # Check if condition is met for 5 consecutive episodes
            print("Stopping early as the average score has been >= 200 for the last 5 episodes.")
            T.save(agent.actor.state_dict(), 'actor.pth')
            break

    # Plotting the results after all games are played
    save_models()  # Save the models before breaking
    episodes = list(range(len(score_history)))
    plot_learning_curve(episodes, score_history, 'performance_plot.png')