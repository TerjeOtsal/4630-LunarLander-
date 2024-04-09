import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque

# Environment
env = gym.make("LunarLanderContinuous-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

# Actor model
def build_actor():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_dim, activation='tanh')
    ])
    return model

# Critic model
def build_critic():
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    
    state_branch = layers.Dense(128, activation='relu')(state_input)
    state_branch = layers.Dense(64, activation='relu')(state_branch)
    
    action_branch = layers.Dense(64, activation='relu')(action_input)
    
    merged = layers.Concatenate()([state_branch, action_branch])
    merged = layers.Dense(32, activation='relu')(merged)
    output = layers.Dense(1)(merged)
    
    model = models.Model(inputs=[state_input, action_input], outputs=output)
    return model

# Noise function for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

    def __call__(self):
        x = (
            self.x_prev 
            + self.theta * (self.mean - self.x_prev) * self.dt 
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# DDPG Agent
class DDPGAgent:
    def __init__(self):
        self.actor = build_actor()
        self.critic = build_critic()
        self.target_actor = build_actor()
        self.target_critic = build_critic()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.actor_optimizer = optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = optimizers.Adam(learning_rate=0.002)
        self.buffer = ReplayBuffer(max_size=1000000)
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=0.2 * np.ones(1))

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add((state, action, reward, next_state, done))

    def train(self, batch_size):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        next_actions = self.target_actor.predict(next_states)
        q_targets_next = self.target_critic.predict([next_states, next_actions])
        q_targets = rewards + 0.99 * q_targets_next * (1 - dones)
        self.critic.train_on_batch([states, actions], q_targets)
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            critic_value = self.critic([states, actions_pred])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.update_target_networks()

    def update_target_networks(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state)[0]
        noise = self.noise()
        action = action + noise
        return np.clip(action, action_low, action_high)

# Initialize agent and training loop
agent = DDPGAgent()
batch_size = 64
episodes = 500

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(500):  # Max episode steps
        print(state)
        state = np.expand_dims(state, axis=0)
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.buffer) > batch_size:
            agent.train(batch_size)

        if done:
            break

    print(f"Episode {episode+1}, Reward: {episode_reward}")

env.close()
