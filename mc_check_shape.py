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

# Plot results for each volatility parameter
start_lag = 0
for param in params_volatility:
    N_lags = param['N_lags']
    end_lag = start_lag + N_lags

    # Calculate theoretical limits
    phi_values = [Phi_Hl(l, H) for l in range(N_lags + 1)]
    phi_values[1] = phi_values[0] + 2*phi_values[1]  # Adjust first value
    factor = nu**2 * theta / (np.sin(np.pi * H) * gamma(2 * H + 1)) * 252 * delta
    theoretical_limit = np.array(phi_values) * factor

    print(aggregated_data[start_lag] / theoretical_limit[1])

    # Plot results
    plt.figure(figsize=(10, 6))
    # plt.bar(range(1,N_lags+1), theoretical_limit[1:], label="Theoretical Limit", alpha=0.6)
    plt.scatter(
        range(1,N_lags+1), 
        theoretical_limit[1:] * aggregated_data[start_lag] / theoretical_limit[1], 
        color="blue", 
        label="Theoretical Limit", zorder=5
    )
    plt.scatter(
        range(1,N_lags+1), 
        aggregated_data[start_lag:end_lag], 
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

    # Update starting lag for the next iteration
    start_lag = end_lag