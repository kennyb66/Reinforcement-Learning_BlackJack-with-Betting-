from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BlackjackAgent:
    def __init__(self, env, learning_rate, epsilon_start, epsilon_final, lambda_decay, discount_factor=0.95):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(3))  # Three possible actions: hit, stand, double down
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.lambda_decay = lambda_decay  # Controls decay speed
        self.episode = 0  # Track episode number
        self.epsilon = epsilon_start
        
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        player_sum, dealer_card, usable_ace = obs

        # Check if double down is allowed
        if player_sum in [10, 11] and dealer_card not in [10, 1] and bankroll >= 2 * bet:
            return 2  # Double Down action

        # Standard ε-greedy action selection
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return int(np.argmax(self.q_values[obs]))  # Exploit best known action


    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """ Exponential decay of epsilon """
        self.epsilon = max(self.epsilon_final, self.epsilon_start * np.exp(-0.000005 * self.episode)) # Logarithmic decay
        self.episode += 1  # Increment episode count


# Hyperparameters
learning_rate = 0.005
n_episodes = 5_000_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

# Initialize environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    epsilon_start=start_epsilon,   
    epsilon_final=final_epsilon,   
    lambda_decay=epsilon_decay,  
)

# Betting implementation
bankroll = 1000  # Starting bankroll
bankroll_list = [bankroll]  # List to store bankroll after each episode

# Win/loss/draw tracking
wins = 0
losses = 0
draws = 0
win_percentage_list = []  # List to store win percentage after each episode

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    # Update default bet size based on best action
    
    default_bet = bankroll / 50  # Default bet size
    bet = default_bet  # Initialize bet size

    while not done:
        action = agent.get_action(obs)  # Choose action

        if action == 2:  # Double Down
            bet *= 2  # Double the bet
            next_obs, reward, terminated, truncated, info = env.step(0)  # Hit once, then stand
            done = True  # Player automatically stands after one hit
        else:
            next_obs, reward, terminated, truncated, info = env.step(action)  # Standard action

        # Update bankroll based on outcome
        if terminated:
            if reward == 1 and action == 2: # Double Down win
                wins += 1
                bankroll += 2 * bet  # Win double bet
            elif reward == 1:
                wins += 1
                bankroll += bet  # Win full bet
            elif reward == -1:
                losses += 1
                bankroll -= bet  # Lose full bet
            else:
                draws += 1

            bankroll_list.append(bankroll)  # Track bankroll

        # Update Q-values using the same reward
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated  # Check if episode is over
        obs = next_obs  # Move to next state

    bet = bankroll / 50  # Reset bet after each round
    if bankroll <= 200:
        bet = 10 # Minimum bet size

    # Calculate win percentage for this episode
    total_games = wins + losses + draws
    if total_games > 0:
        win_percentage = (wins / total_games) * 100 
    else:
        win_percentage = 0
    win_percentage_list.append(win_percentage)  # Store win percentage

    agent.decay_epsilon()  # Reduce exploration

# Print final results
print(f"Final Bankroll: {bankroll}")
#print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
print(f"Win Percentage: {win_percentage_list[-1]:.2f}%")

# visualize the episode rewards, episode length and training error in one figure
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

#np.convolve will compute the rolling mean for 100 episodes

window_size = 100  # Adjust as needed

axs[0].plot(np.convolve(env.return_queue, np.ones(window_size) / window_size, mode="valid"))
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(np.convolve(env.length_queue, np.ones(window_size) / window_size, mode="valid"))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error, np.ones(window_size) / window_size, mode="valid"))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

# Plot bankroll after each episode
#axs[0].plot(bankroll_list)
#axs[0].set_title("Bankroll")
#axs[0].set_xlabel("Episode")
#axs[0].set_ylabel("Bankroll")

plt.tight_layout()
plt.show()

# Save results to a text file after each run
#with open("results.csv", "a") as file:
#    file.write(f"Final Bankroll: {bankroll} \n")