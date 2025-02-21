import numpy as np
import matplotlib.pyplot as plt
from estimator_H import *
import os
from scipy.special import gamma 


params_volatility = [
    {'window': 60, 'N_lags': 12},
    {'window': 90, 'N_lags': 9},
    {'window': 120, 'N_lags': 6},
    {'window': 150, 'N_lags': 4},
]

sampling = 5

delta = sampling/(252 * 23400)

H = 0.5

nu_map = {
    0.1: 0.14175237816573838,
    0.2: 0.23341082890121959,
    0.3: 0.31588951583755265,
    0.4: 0.3874045704067833,
    0.5: 0.45,
}
if H in nu_map:
    nu = nu_map[H]
else:
    raise ValueError(f"Invalid H value: {H}. Must be one of {list(nu_map.keys())}")

theta = 0.02

# Check if file exists and process the data

for param in params_volatility:
    print(theta * nu**2 * (param["window"] * delta) ** (2*H))
