import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(23)
def revised_profit_function(Q, Y, c, p, cS, pL):
    c_tilde = c + cS
    p_tilde = p - pL
    profit = p_tilde * min(Q, Y) - c_tilde * Q
    return profit

def optimal_order_quantity(FY, c_tilde, p_tilde):
    return FY.ppf(p_tilde / (p_tilde + c_tilde))

def monte_carlo_simulation(num_simulations, sample_size, tau, true_demand_dist):
    Q_star = true_demand_dist.ppf(tau)
    Q_np_estimates = []
    Q_p_estimates = []
    
    for _ in range(num_simulations):
        sample = true_demand_dist.rvs(sample_size)
        Q_np = np.quantile(sample, tau)
        Q_p = true_demand_dist.ppf(tau)
        Q_np_estimates.append(Q_np)
        Q_p_estimates.append(Q_p)
    
    return Q_np_estimates, Q_p_estimates

def plot_sensitivity(cS_values, Q_star_values, profit_values):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cS_values, Q_star_values)
    plt.xlabel('Shipping Cost (cS)')
    plt.ylabel('Optimal Order Quantity (Q*)')
    plt.title('Sensitivity of Q* to Shipping Cost')

    plt.subplot(1, 2, 2)
    plt.plot(cS_values, profit_values)
    plt.xlabel('Shipping Cost (cS)')
    plt.ylabel('Optimal Expected Profit')
    plt.title('Sensitivity of Profit to Shipping Cost')

    plt.tight_layout()
    plt.show()

def bootstrap_nonparametric_estimator(data, tau, num_bootstrap_samples):
    n = len(data)
    bootstrap_estimates = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        Q_np = np.quantile(bootstrap_sample, tau)
        bootstrap_estimates.append(Q_np)

    bootstrap_estimates = np.array(bootstrap_estimates)
    return bootstrap_estimates

def calculate_bootstrap_statistics(bootstrap_estimates):
    estimate_mean = np.mean(bootstrap_estimates)
    estimate_std_error = np.std(bootstrap_estimates)
    confidence_interval = np.percentile(bootstrap_estimates, [2.5, 97.5])
    return estimate_mean, estimate_std_error, confidence_interval

def monte_carlo_bootstrap_simulation(num_simulations, sample_size, tau, true_demand_dist, num_bootstrap_samples):
    Q_star = true_demand_dist.ppf(tau)
    bootstrap_results = []

    for _ in range(num_simulations):
        sample = true_demand_dist.rvs(sample_size)
        bootstrap_estimates = bootstrap_nonparametric_estimator(sample, tau, num_bootstrap_samples)
        mean_estimate, _, _ = calculate_bootstrap_statistics(bootstrap_estimates)
        bootstrap_results.append(mean_estimate)
    
    return bootstrap_results

data = pd.read_excel('/mnt/data/BakeryData2024_Vilnius.xlsx')

product_sales = data['main street A'].dropna()

num_bootstrap_samples = 1000
tau = 0.9

bootstrap_estimates = bootstrap_nonparametric_estimator(product_sales, tau, num_bootstrap_samples)
mean_estimate, std_error, conf_interval = calculate_bootstrap_statistics(bootstrap_estimates)

print(f"Bootstrap Mean Estimate for 'main street A': {mean_estimate}")
print(f"Bootstrap Standard Error for 'main street A': {std_error}")
print(f"Bootstrap 95% Confidence Interval for 'main street A': {conf_interval}")

num_simulations = 100
sample_size = len(product_sales)
true_demand_dist = norm(loc=50, scale=10)

bootstrap_results = monte_carlo_bootstrap_simulation(num_simulations, sample_size, tau, true_demand_dist, num_bootstrap_samples)
mean_bootstrap_result = np.mean(bootstrap_results)
std_error_bootstrap_result = np.std(bootstrap_results)
confidence_interval_bootstrap_result = np.percentile(bootstrap_results, [2.5, 97.5])

print(f"Monte Carlo Bootstrap Mean Estimate: {mean_bootstrap_result}")
print(f"Monte Carlo Bootstrap Standard Error: {std_error_bootstrap_result}")
print(f"Monte Carlo Bootstrap 95% Confidence Interval: {confidence_interval_bootstrap_result}")
