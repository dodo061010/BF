import numpy as np
import matplotlib.pyplot as plt

mu = np.array([0.05, 0.15, 0.2])
Sigma = np.array([[0.1, 0, 0.15], [0, 0.2, 0], [0.15, 0, 0.3]])


def portfolio_variances(weights_matrix, sigma):
    """Compute portfolio variances for multiple portfolios."""
    num_portfolios = weights_matrix.shape[0]
    variances = np.zeros(num_portfolios)

    for p in range(num_portfolios):
        weights = weights_matrix[p]
        variance = 0
        for i in range(len(weights)):
            for j in range(len(weights)):
                variance += weights[i] * weights[j] * sigma[i, j]
        variances[p] = variance

    return variances


# 1.a
np.random.seed(123)
num_portfolios = 1000

raw_weights = np.random.rand(num_portfolios, 3)

row_sums = raw_weights.sum(axis=1)
weights = raw_weights / row_sums[:, np.newaxis]

portfolio_returns = np.dot(weights, mu)
portfolio_std_dev = np.sqrt(portfolio_variances(weights, Sigma))

# Scatterplot for the Markowitz bullet
plt.figure(figsize=(10, 7))
plt.scatter(portfolio_std_dev, portfolio_returns, c="blue", marker="o")
plt.title("Markowitz Bullet with 3 Assets")
plt.xlabel("Portfolio Standard Deviation")
plt.ylabel("Portfolio Expected Return")
plt.grid(True)
plt.savefig("Markowitz Bullet with 3 Assets.png")

# Adjusting the covariance matrix
Sigma_modified = Sigma.copy()
Sigma_modified[0, 2] = 0
Sigma_modified[2, 0] = 0

# Calculate expected returns and standard deviations of portfolios with modified covariance matrix
portfolio_std_dev_modified = np.sqrt(portfolio_variances(weights, Sigma_modified))

# Scatterplot for the modified Markowitz bullet
plt.figure(figsize=(10, 7))
plt.scatter(
    portfolio_std_dev, portfolio_returns, c="blue", marker="o", label="Original"
)
plt.scatter(
    portfolio_std_dev_modified,
    portfolio_returns,
    c="red",
    marker="o",
    label="Modified (σ13 = 0)",
)
plt.title("Comparison of Original and Modified Markowitz Bullets")
plt.xlabel("Portfolio Standard Deviation")
plt.ylabel("Portfolio Expected Return")
plt.legend()
plt.grid(True)
plt.savefig("Comparison of Original and Modified Markowitz Bullets.png")
