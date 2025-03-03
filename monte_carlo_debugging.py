import numpy as np
import matplotlib.pyplot as plt
from lib.estimator_H import *
import os
from scipy.special import gamma 
import pandas as pd


params_volatility = [
    {'window': 60, 'N_lags': 12},
    {'window': 90, 'N_lags': 9},
    {'window': 120, 'N_lags': 6},
    {'window': 150, 'N_lags': 4},
]

plot_res = False

sampling = 5

delta = sampling/(252 * 23400)

H = 0.1
identificator = '5s'

folder_name = "/Users/gregoire.szymanski/Documents/mc_results"
filename = f"results{int(H * 10):02d}_{identificator}.txt"
file_path = os.path.join(folder_name, filename)

print(file_path)


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
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    try:
        # Parse data
        data = [np.array([float(value) for value in line.split(",")]) for line in lines]
    except ValueError:
        raise ValueError("Error processing file data. Ensure file contains numeric values.")
    
    if len(data) == 0:
        raise ValueError("File is empty or contains no valid data.")
    
    aggregated_data = np.mean(data, axis=0)
else:
    raise FileNotFoundError(f"File not found: {file_path}")


param = params_volatility[0]
start_lag = 0
end_lag = 12
N_lags = 12
data = aggregated_data[start_lag:end_lag]

def Psi(Hpsi):
    p = []
    window = param['window']
    N_lags = param['N_lags']
    factor = window**(2 * Hpsi)
    p.append(factor * (Phi_Hl(0, Hpsi) + 2 * Phi_Hl(1, Hpsi)))
    for i in range(2, N_lags + 1):
        p.append(factor * Phi_Hl(i, Hpsi))
    return np.array(p)

data = data / data[0]

identity = np.identity(len(data))
H_total_id, _ = estimation_GMM(identity,
                                data,
                                Psi,
                                0.001,
                                0.499,
                                0.0001)

print(H_total_id)

print(data)

R_the = F_GMM_get_R(identity, data, Psi, 0.1)
R_est = F_GMM_get_R(identity, data, Psi, H_total_id)

# Calculate theoretical limits
phi_values = [Phi_Hl(l, H) for l in range(13)]
phi_values[1] = phi_values[0] + 2*phi_values[1] 
theoretical_limit = np.array(phi_values)[1:]

# Calculate theoretical limits
phi_values = [Phi_Hl(l, H_total_id) for l in range(13)]
phi_values[1] = phi_values[0] + 2*phi_values[1] 
theoretical_limit_est = np.array(phi_values)[1:]



plt.figure(figsize=(10, 6))
plt.scatter(
    range(1,N_lags+1), 
    theoretical_limit * R_the, 
    color="green", 
    label="Theoretical Limit", zorder=5
)
plt.scatter(
    range(1,N_lags+1), 
    theoretical_limit_est * R_est, 
    color="blue", 
    label="Estimated Limit", zorder=5
)
plt.scatter(
    range(1,N_lags+1), 
    data, 
    color="red", 
    label="Aggregated Data",
    zorder=5
)
plt.xlabel("Lag Index")
plt.ylabel("Value")
plt.title(f"Comparison of Theoretical Limit and Aggregated Data (Window: {param['window']})")
plt.legend()
plt.grid(True)
plt.show()

