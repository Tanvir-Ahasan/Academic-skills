import numpy as np
import scipy.stats as stats
import multiprocessing as mp

# Set the seed for reproducibility
np.random.seed(21)

# Parameters for the distributions
mu_norm, sigma_norm = 115, 10       # Parameters for the normal distribution
mu_lognorm, sigma_lognorm = 6, 0.6  # Parameters for the log-normal distribution
n_samples = [10, 50, 100, 200]      # Different sample sizes to test
target_service_levels = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # Target service levels
M = 1000  # Number of Monte Carlo repetitions

# Define the nonparametric estimator function
def nonparametric(Y, tau):
    return np.quantile(Y, tau)

# Define the parametric estimator function
def parametric(Y, tau, distribution='normal'):
    if distribution == 'normal':
        mu_hat, sigma_hat = np.mean(Y), np.std(Y)
        return stats.norm.ppf(tau, loc=mu_hat, scale=sigma_hat)
    elif distribution == 'lognormal':
        shape, loc, scale = stats.lognorm.fit(Y, floc=0)
        return stats.lognorm.ppf(tau, shape, loc=loc, scale=scale)

# Define the profit function
def profit(Q, Y, c, p):
    return p * np.minimum(Q, Y) - c * Q

# Function to run the simulation for a single combination of parameters
def run_simulation(distribution, n, tau, mu_norm, sigma_norm, mu_lognorm, sigma_lognorm, M):
    results = []
    
    # True order quantity based on the target service level
    if distribution == 'normal':
        target_order_quantity = stats.norm.ppf(tau, loc=mu_norm, scale=sigma_norm)
    else:
        target_order_quantity = stats.lognorm.ppf(tau, sigma_lognorm, scale=np.exp(mu_lognorm))
    
    nonparametric_orders = []
    parametric_orders = []
    nonparametric_profits = []
    parametric_profits = []

    for _ in range(M):
        # Generate synthetic demand data
        if distribution == 'normal':
            Y = np.random.normal(mu_norm, sigma_norm, n)
        elif distribution == 'lognormal':
            Y = np.random.lognormal(mu_lognorm, sigma_lognorm, n)
        
        # Estimate order quantities
        Q_nonparametric = nonparametric(Y, tau)
        Q_parametric = parametric(Y, tau, distribution)
        
        # Actual demand for profit calculation
        Y_obs = np.random.choice(Y)
        
        # Record estimated order quantities
        nonparametric_orders.append(Q_nonparametric)
        parametric_orders.append(Q_parametric)
        
        # Calculate profits
        profit_nonparametric = profit(Q_nonparametric, Y_obs, 1-tau, 1)
        profit_parametric = profit(Q_parametric, Y_obs, 1-tau, 1)
        
        nonparametric_profits.append(profit_nonparametric)
        parametric_profits.append(profit_parametric)
    
    # Calculate performance metrics
    nonparametric_RMSE = np.sqrt(np.mean((np.array(nonparametric_orders) - target_order_quantity) ** 2))
    parametric_RMSE = np.sqrt(np.mean((np.array(parametric_orders) - target_order_quantity) ** 2))
    RMSE_comparison_ratio = nonparametric_RMSE / parametric_RMSE
    
    nonparametric_SL = np.mean(np.array(nonparametric_orders) >= Y_obs)
    parametric_SL = np.mean(np.array(parametric_orders) >= Y_obs)
    
    R_star = profit(target_order_quantity, Y_obs, 1-tau, 1)
    nonparametric_PLR = np.mean(np.abs(
    
