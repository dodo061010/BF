import numpy as np
import matplotlib.pyplot as plt

mu = np.array([0.05, 0.15, 0.2])
Sigma = np.array([[0.1, 0, 0.15], [0, 0.2, 0], [0.15, 0, 0.3]])

# 1.a
np.random.seed(123)
num_portfolios = 1000

raw_weights = np.random.rand(num_portfolios, 3)

row_sums = raw_weights.sum(axis=1)
weights = raw_weights / row_sums[:, np.newaxis]

portfolio_returns = np.dot(weights, mu)
