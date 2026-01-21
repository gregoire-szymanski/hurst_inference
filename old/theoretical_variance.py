from lib.estimator_H import *

delta = 5.0 / (252.0 * 23400)   # Time increment
n_days = 1

t = n_days / 252 

params_volatility = [
    {'window': 120, 'N_lags': 6},
    {'window': 150, 'N_lags': 4}
]

H_var, _ = get_optimal_variance(params_volatility, 0.1, 0.1, t, delta)
print(H_var)
H_var, _ = get_optimal_variance(params_volatility, 0.1, 0.2, t, delta)
print(H_var)
H_var, _ = get_optimal_variance(params_volatility, 0.1, 0.3, t, delta)
print(H_var)

exit()

H_var, _ = get_optimal_variance(params_volatility, 0.2, 0.23, t, delta)
print(H_var)

H_var, _ = get_optimal_variance(params_volatility, 0.3, 0.31, t, delta)
print(H_var)

H_var, _ = get_optimal_variance(params_volatility, 0.4, 0.39, t, delta)
print(H_var)

H_var, _ = get_optimal_variance(params_volatility, 0.5, 0.45, t, delta)
print(H_var)

