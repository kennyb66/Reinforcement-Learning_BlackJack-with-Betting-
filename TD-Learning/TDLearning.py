from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BlackjackAgent:
    def __init__(self, env, learning_rate, epsilon_start, epsilon_final, decay_rate, discount_factor=0.95):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(2))  # Two actions: hit (0), stand (1)
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.decay_rate = decay_rate
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
    ):
        """Updates Q-value using TD(0) update rule."""
        # Get the maximum Q-value for the next state
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Logarithmic decay of epsilon."""
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * (1 / (1 + self.decay_rate * self.episode))
        self.epsilon = max(self.epsilon, self.epsilon_final)  # Ensure epsilon doesn't go below final_epsilon
        self.episode += 1

# Hyperparameters
learning_rate = 0.01
n_episodes = 10_000_000
start_epsilon = 1.0
final_epsilon = 0.001
decay_rate = 0.0001 

# Initialize environment and agent
env = gym.make("Blackjack-v1", sab=True)
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

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Track wins, losses, draws
        if terminated:
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

        # TD(0) update
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

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