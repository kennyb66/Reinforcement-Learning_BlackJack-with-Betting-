from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BlackjackAgent:
    def __init__(self, env, learning_rate, epsilon_start, epsilon_final, lambda_decay, discount_factor=0.95):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(3))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.lambda_decay = lambda_decay
        self.episode = 0
        self.epsilon = epsilon_start
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        player_sum, dealer_card, usable_ace = obs
        total_games = wins + losses + draws
        
        # Only allow double down after 5M games when bankroll tracking begins
        if total_games > 8_000_000 and player_sum in [10, 11] and dealer_card not in [10, 1] and bankroll >= 2 * bet:
            return 2
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool]):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon_start * np.exp(-0.000005 * self.episode))
        self.episode += 1


# Hyperparameters
learning_rate = 0.01
n_episodes = 10_000_000  
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.001

# Initialize environment and agent
env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    epsilon_start=start_epsilon,
    epsilon_final=final_epsilon,
    lambda_decay=epsilon_decay,
)

# Betting implementation (novelty)
bankroll = 10000
bankroll_list = []
wins = 0
losses = 0
draws = 0
win_percentage_list = []

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    total_games = wins + losses + draws
    default_bet = 20  # Fixed bet size until 5M games
    bet = default_bet

    # Start using dynamic bankroll after 5M games
    if total_games > 8_000_000:
        default_bet = bankroll / 50
        bet = default_bet

    while not done:
        action = agent.get_action(obs)
        
        if action == 2:
            bet *= 2
            next_obs, reward, terminated, truncated, info = env.step(0)
            done = True
        else:
            next_obs, reward, terminated, truncated, info = env.step(action)

        # Update bankroll only after 8M games
        if terminated and total_games > 8_000_000:
            if reward == 1 and action == 2:
                wins += 1
                bankroll += 2 * bet
            elif reward == 1:
                wins += 1
                bankroll += bet
            elif reward == -1:
                losses += 1
                bankroll -= bet
            else:
                draws += 1
            bankroll_list.append(bankroll)

        # Always count wins/losses/draws, but only for win % after 8M
        if terminated:
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs

    # Adjust bet after 8M games
    if total_games > 8_000_000:
        bet = bankroll / 50
        if bankroll <= 200:
            bet = 10

    # Calculate win percentage only after 8M games
    total_games = wins + losses + draws
    if total_games > 5_000_000:
        win_percentage = (wins / total_games) * 100
        win_percentage_list.append(win_percentage)
    else:
        win_percentage_list.append(0)  # Append 0 before 8M games

    agent.decay_epsilon()

# Print final results
print(f"Final Bankroll: {bankroll}")
print(f"Win Percentage: {win_percentage_list[-1]:.2f}%")

# Separate plots
window_size = 100

# Plot 1: Episode Rewards
plt.figure(figsize=(10, 6))
plt.plot(np.convolve(env.return_queue, np.ones(window_size) / window_size, mode="valid"))
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Plot 2: Bankroll
plt.figure(figsize=(10, 6))
plt.plot(bankroll_list)
plt.title("Bankroll")
plt.xlabel("Episode after 5M")
plt.ylabel("Bankroll")
plt.show()

# Plot 3: Training Error
plt.figure(figsize=(10, 6))
plt.plot(np.convolve(agent.training_error, np.ones(window_size) / window_size, mode="valid"))
plt.title("Training Error")
plt.xlabel("Episode")
plt.ylabel("Temporal Difference")
plt.show()

# Plot 4: Win Percentage
plt.figure(figsize=(10, 6))
plt.plot(win_percentage_list)
plt.title("Win Percentage")
plt.xlabel("Episode")
plt.ylabel("Win Percentage (%)")
plt.show()