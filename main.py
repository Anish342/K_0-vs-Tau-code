import numpy as np
from scipy.special import j0, j1
import matplotlib.pyplot as plt

alpha = 1
beta = 0.01


# Newton-Raphson method to find mu_n of the equation
def calculate_mu_n(beta):
    guess = beta  # Initial guess for mu_n
    tolerance = 1e-5
    max_iterations = 1000
    for _ in range(max_iterations):
        f = guess * j1(guess) - beta * j0(guess)
        if abs(f) < tolerance:
            return guess
        f_prime = j1(guess) + guess * (j0(guess) - (j1(guess) / guess)) + beta * j1(guess)
        guess = guess - f / f_prime
    return None


# Calculate An
def calculate_An(mu_n, a, b):
    return (2 * mu_n * j1(mu_n * a)) / (a * (mu_n ** 2 + b ** 2) * j0(mu_n) ** 2)


# Calculate K0
def calculate_K0(tau, mu_n_values, An_values):
    numerator = -np.sum(np.array(An_values) * np.array(mu_n_values) * np.array(j1(mu_n_values)) * np.exp(
        (-np.array(mu_n_values) ** 2) * tau))
    denominator = np.sum((np.array(An_values) / np.array(mu_n_values)) * np.array(j1(mu_n_values)) * np.exp(
        (-np.array(mu_n_values) ** 2) * tau))
    return numerator / denominator


# Find mu_n using Newton-Raphson method
roots = [calculate_mu_n(b) for b in np.linspace(1e-2, 10000, 1000) if calculate_mu_n(b) is not None]

# Calculate An for each mu_n
An_values = [calculate_An(root, alpha, beta) for root in roots]

# Define tau values
tau_values = np.linspace(1e-4, 1, 100)

# Calculate K0 for each tau value
K0_values = [calculate_K0(tau, roots, An_values) for tau in tau_values]
minus_K0_values = [-K0 for K0 in K0_values]

# Plot K0 against mu_n
plt.plot(tau_values, minus_K0_values, label='K0 vs tau')
plt.xlabel('tau')
plt.ylabel('-K0')
plt.title('K0 vs tau\n')
plt.xlim(1e-2, 1)
plt.ylim(1e-4, 1e-1)
# print alpha, beta on plot
plt.text(0.5, 0.1, f'alpha = {alpha}, beta = {beta}', fontsize=12, ha='center')
plt.grid(True)
plt.show()
