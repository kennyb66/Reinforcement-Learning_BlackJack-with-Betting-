from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BlackjackAgent:
    def __init__(self, env, learning_rate, epsilon_start, epsilon_final, lambda_decay):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(3))  # Q-values for 3 actions
        self.returns_sum = defaultdict(float)  # Sum of returns for averaging
        self.returns_count = defaultdict(int)  # Count of visits for averaging
        self.lr = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.lambda_decay = lambda_decay
        self.episode = 0
        self.epsilon = epsilon_start

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        player_sum, dealer_card, usable_ace = obs
        total_games = wins + losses + draws
        if total_games > 8_000_000 and player_sum in [10, 11] and dealer_card not in [10, 1] and bankroll >= 2 * bet:
            return 2
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

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

# Betting implementation
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
    default_bet = 20
    bet = default_bet
    episode_history = []  # Store (state, action, reward) for Monte Carlo

    if total_games > 8_000_000:
        default_bet = bankroll / 50
        bet = default_bet

    # Play episode and store history
    while not done:
        action = agent.get_action(obs)
        if action == 2:
            bet *= 2
            next_obs, reward, terminated, truncated, info = env.step(0)
            done = True
        else:
            next_obs, reward, terminated, truncated, info = env.step(action)

        # Store step in episode history
        episode_history.append((obs, action, reward))

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

        if terminated:
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

        done = terminated or truncated
        obs = next_obs

    # Adjust bet after 8M games
    if total_games > 8_000_000:
        bet = bankroll / 50
        if bankroll <= 200:
            bet = 10

    # Calculate win percentage only after 8M games
    total_games = wins + losses + draws
    if total_games > 8_000_000:
        win_percentage = (wins / total_games) * 100
        win_percentage_list.append(win_percentage)
    else:
        win_percentage_list.append(0)

    # Monte Carlo update after episode
    visited = set()  # For first-visit MC
    total_return = 0
    for t in reversed(range(len(episode_history))):
        state, action, reward = episode_history[t]
        total_return += reward  # Accumulate return (undiscounted)
        state_action = (state, action)
        
        if state_action not in visited:  # First-visit MC
            visited.add(state_action)
            agent.returns_sum[state_action] += total_return
            agent.returns_count[state_action] += 1
            # Update Q-value incrementally
            agent.q_values[state][action] = agent.returns_sum[state_action] / agent.returns_count[state_action]

    agent.decay_epsilon()

# Print final results
print(f"Final Bankroll: {bankroll}")
print(f"Win Percentage: {win_percentage_list[-1]:.2f}%")

# Separate plots
window_size = 100

plt.figure(figsize=(10, 6))
plt.plot(np.convolve(env.return_queue, np.ones(window_size) / window_size, mode="valid"))
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(bankroll_list)
plt.title("Bankroll")
plt.xlabel("Episode after 8M")
plt.ylabel("Bankroll")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(win_percentage_list)
plt.title("Win Percentage")
plt.xlabel("Episode")
plt.ylabel("Win Percentage (%)")
plt.show()