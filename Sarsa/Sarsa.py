from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BlackjackAgent:
    def __init__(self, env, learning_rate, epsilon_start, epsilon_final, decay_rate, discount_factor=0.95):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(2))  # Two actions: hit (0), stand (1) - removed double down
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.decay_rate = decay_rate  # Linear decay rate
        self.episode = 0
        self.epsilon = epsilon_start
        
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Epsilon-greedy action selection for hit (0) or stand (1)."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return int(np.argmax(self.q_values[obs]))  # Exploit

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
        next_action: int,
    ):
        """Updates Q-value using SARSA update rule."""
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Linear decay of epsilon"""
        self.epsilon = max(self.epsilon_final, self.epsilon_start - self.decay_rate * self.episode)
        self.episode += 1

# Hyperparameters
learning_rate = 0.005
n_episodes = 5_000_000
start_epsilon = 1.0
final_epsilon = 0.1
decay_rate = (start_epsilon - final_epsilon) / (n_episodes / 2)  # Linear decay over half the episodes

# Initialize environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    epsilon_start=start_epsilon,
    epsilon_final=final_epsilon,
    decay_rate=decay_rate,
)

# Win/loss/draw tracking
wins = 0
losses = 0
draws = 0

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # Get initial action
    action = agent.get_action(obs)

    while not done:
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Get next action for SARSA
        next_action = agent.get_action(next_obs)

        # Track wins, losses, draws
        if terminated:
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

        # SARSA update
        agent.update(obs, action, reward, terminated, next_obs, next_action)

        done = terminated or truncated
        obs = next_obs
        action = next_action

    agent.decay_epsilon()

# Print results
total_games = wins + losses + draws
win_percentage = (wins / total_games * 100) if total_games > 0 else 0
print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
print(f"Win Percentage: {win_percentage:.2f}%")

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
window_size = 100

axs[0].plot(np.convolve(env.return_queue, np.ones(window_size) / window_size, mode="valid"))
axs[0].set_title("Rewards of each Episode")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(np.convolve(env.length_queue, np.ones(window_size) / window_size, mode="valid"))
axs[1].set_title("The Length of Each Episode")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error, np.ones(window_size) / window_size, mode="valid"))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout()
plt.show()