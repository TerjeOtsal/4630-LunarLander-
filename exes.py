import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch as T
import matplotlib.pyplot as plt
import gym

class OUActionNoise():
    def __init__(self, mean, sigma=0.15, theta=0.2, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class Buffer():
    def __init__(self, buffer_capacity, input_shape, n_actions):
        self.mem_size = buffer_capacity
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.mean = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mean(x))

        return x


# Define Agent class
class DDPG_Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 buffer_capacity=500000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(buffer_capacity, input_dims, n_actions)

        self.noise = OUActionNoise(mean=np.zeros(n_actions))

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
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float) 
        mean = self.actor.forward(state) 
        mean_prime = mean + T.tensor(self.noise(),
                                    dtype=T.float)
        self.actor.train()

        return mean_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float)
        next_states = T.tensor(next_states, dtype=T.float)  
        actions = T.tensor(actions, dtype=T.float)  
        rewards = T.tensor(rewards, dtype=T.float)  
        dones = T.tensor(dones)  

        target_actions = self.target_actor.forward(next_states)
        critic_value_ = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[dones] = 0.0
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


#LunarLander environment
env = gym.make('LunarLander-v2', continuous=True, gravity=np.random.uniform(-12, 0), enable_wind=True, wind_power=np.random.uniform(0, 20), turbulence_power=np.random.uniform(0, 2))

# Initializ the agent
agent = Agent(alpha=0.0005, beta=0.005, input_dims=env.observation_space.shape, tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=env.action_space.shape[0])

# Training loop
episodes = 1000
score_history = []
avg_score_history = []

for i in range(episodes):
    # Reset environment and episode variables
    observation, _ = env.reset()
    done = False
    score = 0
    agent.noise.reset()

    # Episode loop
    while not done:
        # Choose action, take step, and learn
        action = agent.choose_action(observation)
        next_observation, reward, done, info, _ = env.step(action)
        agent.remember(observation, action, reward, next_observation, done)
        agent.learn()

        # Update episode score and observation
        score += reward
        observation = next_observation

    # Append episode score and calculate average score
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_score_history.append(avg_score)

    # Print episode information
    print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)

    # Plot average score every 10 episodes
    if (i + 1) % 10 == 0:
        plt.plot(avg_score_history)
        plt.xlabel('Last 100 Episodes (x100)')
        plt.ylabel('Average Score')
        plt.title('Average Score After Each 100 Episodes')
        plt.axhline(y=200, color='r', linestyle='-', label='Successful Landing Threshold (200)')
        plt.legend()
        plt.show()