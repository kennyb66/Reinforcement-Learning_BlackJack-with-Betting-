import matplotlib.pyplot as plt
import numpy as np
import csv

# Load the CSV file
csv_file = "betting_results.csv"  # Update this to your actual filename

# Initialize storage lists
bankrolls = []

# Read CSV and extract values
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        if row:  # Ensure row is not empty
            # Extract numerical values
            try:
                bankroll = float(row[0].split(":")[1].split(",")[0].strip())  # Extract bankroll
                
                # Append to lists
                bankrolls.append(bankroll)

            except Exception as e:
                print(f"Skipping invalid row: {row} | Error: {e}")

# Convert lists to NumPy arrays
bankrolls = np.array(bankrolls)

# Create singular plot
fig, axs = plt.subplots(1, 1, figsize=(14, 6))

# Histogram of Final Bankrolls
axs.hist(bankrolls, bins=15, color="blue", edgecolor="black", alpha=0.7)
axs.axvline(np.mean(bankrolls), color="red", linestyle="dashed", linewidth=2, label=f"Mean: ${np.mean(bankrolls):.2f}")
axs.set_title("Distribution of Final Bankrolls")
axs.set_xlabel("Final Bankroll ($)")
axs.set_ylabel("Frequency")
axs.legend()

# Show plots
plt.tight_layout()
plt.show()
