from parameters import *
from lib.estimator_H import *

n_days = 252

H_var, _ = get_optimal_variance(params_volatility, 0.1, 0.1, n_days, delta)
print(H_var)

H_var, _ = get_optimal_variance(params_volatility, 0.3, 0.22, n_days, delta)
print(H_var)

H_var, _ = get_optimal_variance(params_volatility, 0.5, 0.45, n_days, delta)
print(H_var)

