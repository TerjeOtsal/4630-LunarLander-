import torch as T
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

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

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = None  # Not needed for testing
        self.noise = None  # Not used during testing
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions)
        self.actor.to(self.actor.device)

    def load_models(self):
        self.actor.load_state_dict(T.load('actor.pth'))

    def choose_action(self, observation):
        self.actor.eval()
        if isinstance(observation, np.ndarray) and observation.ndim == 1:
            observation = observation.reshape(1, -1)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        return mu.cpu().detach().numpy()[0]

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', continuous=True)
    agent = Agent(alpha=0.0002, beta=0.002, 
                  input_dims=env.observation_space.shape, tau=0.005,
                  batch_size=96, fc1_dims=600, fc2_dims=450, 
                  n_actions=env.action_space.shape[0])

    agent.load_models()

    n_games = 10
    scores = []

    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()  # Unpack the tuple correctly
        print(f'Initial observation: {observation}')
        print(f'Observation shape: {observation.shape}')
        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, _, info = env.step(action)  # Unpack the tuple correctly
            print(f'New observation: {observation}')
            print(f'Observation shape: {observation.shape}')
            score += reward
        scores.append(score)
        print(f'Episode {i+1}: Score = {score}')

    env.close()

    # Plotting the scores
    plt.plot(scores, label='Score per Episode')
    plt.title('Performance of Trained Model')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()