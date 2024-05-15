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

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # First fully connected layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Second fully connected layer

        self.bn1 = nn.LayerNorm(self.fc1_dims)  # Layer normalization after first layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)  # Layer normalization after second layer

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)  # Output layer for actions

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)  # Optimizer for the network
        
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')  # Set device to GPU if available, else CPU

    def forward(self, state):
        x = self.fc1(state)  # Forward pass through first layer
        x = self.bn1(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Forward pass through second layer
        x = self.bn2(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation
        x = T.tanh(self.mu(x))  # Final output with Tanh activation

        return x

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64):
        self.gamma = gamma  # Discount factor for future rewards
        self.tau = tau  # Soft update parameter
        self.memory = None  # Not needed for testing
        self.noise = None  # Not used during testing
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions)  # Actor network
        self.actor.to(self.actor.device)  # Move actor network to the correct device

    def load_models(self):
        self.actor.load_state_dict(T.load('../SavedModel/actor.pth'))  # Load pre-trained model weights

    def choose_action(self, observation):
        self.actor.eval()  # Set network to evaluation mode
        if isinstance(observation, np.ndarray) and observation.ndim == 1:
            observation = observation.reshape(1, -1)  # Reshape observation if it's a 1D array
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)  # Convert observation to tensor and move to device
        mu = self.actor.forward(state).to(self.actor.device)  # Get action from actor network
        return mu.cpu().detach().numpy()[0]  # Convert action tensor to numpy array and return

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', continuous=True)  # Initialize environment
    agent = Agent(alpha=0.0002, beta=0.002, 
                  input_dims=env.observation_space.shape, tau=0.005,
                  batch_size=96, fc1_dims=600, fc2_dims=450, 
                  n_actions=env.action_space.shape[0])  # Initialize agent

    agent.load_models()  # Load pre-trained models

    n_games = 50  # Number of episodes
    scores = []  # List to store scores

    for i in range(n_games):
        score = 0  # Initialize score for the episode
        done = False  # Initialize done flag
        observation, info = env.reset()  # Unpack the tuple correctly
        print(f'Initial observation: {observation}')  # Print initial observation
        print(f'Observation shape: {observation.shape}')  # Print shape of initial observation
        while not done:
            action = agent.choose_action(observation)  # Choose action based on current observation
            observation, reward, done, _, info = env.step(action)  # Unpack the tuple correctly
            print(f'New observation: {observation}')  # Print new observation
            print(f'Observation shape: {observation.shape}')  # Print shape of new observation
            score += reward  # Accumulate reward
        scores.append(score)  # Append score for the episode
        print(f'Episode {i+1}: Score = {score}')  # Print episode score

    env.close()  # Close the environment

     # Plotting the scores
    plt.plot(scores, label='Score per Episode')  # Plot scores
    plt.title('Performance of Trained Model')  # Plot title
    plt.xlabel('Episode')  # X-axis label
    plt.ylabel('Score')  # Y-axis label
    plt.legend()  # Show legend
    plt.savefig('trained_model_results.png')  # Save the plot as an image
    plt.show()  # Display plot