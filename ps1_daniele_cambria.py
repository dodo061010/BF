import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

"""
Python 3.10.8

Requirements:
matplotlib==3.7.3
numpy==1.25.2
pandas==2.1.1
scipy==1.11.3
"""

# Create ./plots in working directory if it doesn't exist
if not os.path.exists("./plots"):
    os.makedirs("./plots")

# ======================= 5a =======================

# Step 1: Generate the dataset
dice_data = np.random.choice(range(1, 7), size=(1000, 10))
dice_df = pd.DataFrame(dice_data, columns=[f"X{i+1}" for i in range(10)])

# Step 2: Plot histograms for dice rolls
# Adjusting histograms for dice rolls with proper bin edges
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
axes = axes.ravel()

for i in range(10):
    cumulative_sum = dice_df.iloc[:, : i + 1].sum(axis=1)
    bin_edges = np.arange(cumulative_sum.min() - 0.5, cumulative_sum.max() + 1.5)
    axes[i].hist(cumulative_sum, bins=bin_edges, density=True, alpha=0.7, edgecolor="k")
    axes[i].set_title(f"Histogram of $X_1 + \ldots + X_{i+1}$")
    axes[i].set_xlim([cumulative_sum.min() - 1, cumulative_sum.max() + 1])

plt.tight_layout()
plt.savefig("./plots/dice_histograms.png")
plt.show()

# Step 3: Generate normally distributed data and plot histograms
normal_data = np.random.randn(1000, 10)
normal_df = pd.DataFrame(normal_data, columns=[f"Normal_X{i+1}" for i in range(10)])

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
axes = axes.ravel()

for i in range(10):
    cumulative_sum = normal_df.iloc[:, : i + 1].sum(axis=1)
    axes[i].hist(cumulative_sum, bins=50, density=True, alpha=0.7, edgecolor="k")
    axes[i].set_title(f"Histogram of Normal $X_1 + \ldots + X_{i+1}$")

plt.tight_layout()
plt.savefig("./plots/normal_histograms.png")
plt.show()


# ======================= 5b =======================

# Given values
mean_1, std_1 = 0.02, 0.06  # Asset 1
mean_2, std_2 = 0.08, 0.08  # Asset 2
weight_1, weight_2 = 0.5, 0.5  # Portfolio weights

# Portfolio mean and standard deviation
portfolio_mean = weight_1 * mean_1 + weight_2 * mean_2
portfolio_variance = (weight_1**2 * std_1**2) + (
    weight_2**2 * std_2**2
)  # Since assets are uncorrelated
portfolio_std = np.sqrt(portfolio_variance)

# Probability of negative return
probability_negative_return = norm.cdf(0, loc=portfolio_mean, scale=portfolio_std)

# Given z-score for the 5th percentile
z_score_5th_percentile = -1.64

# Calculate VaR using the provided z-score
VaR_5th_percentile = portfolio_mean + z_score_5th_percentile * portfolio_std

print(
    f"Probability of negative return: {probability_negative_return}, Value at Risk at the 5th percentile: {VaR_5th_percentile}"
)

# ======================= 6 =======================

# Step 1: Simulate 1000 draws of a random normal variable with mean zero and variance 1
data = np.random.randn(1000)

# Step 2: Plot the histogram (empirical PDF)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7, edgecolor="k")
plt.title("Histogram (Empirical PDF) of Simulated Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("./plots/empirical_pdf.png")
plt.show()

# Step 3: Plot the empirical CDF
sorted_data = np.sort(data)
percentile_ranks = np.arange(1, len(data) + 1) / len(data)

plt.figure(figsize=(10, 6))
plt.plot(sorted_data, percentile_ranks, marker=".", linestyle="none")
plt.title("Empirical CDF of Simulated Data")
plt.xlabel("Value")
plt.ylabel("Percentile Rank")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("./plots/empirical_cdf.png")
plt.show()
