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
    
    data_1 = data[:31500]
    data_2 = data[31500:31500*2]
    data_3 = data[31500*2:31500*3]
    data_4 = data[31500*3:126000]

    aggregated_data_1 = np.mean(data_1, axis=0)
    aggregated_data_2 = np.mean(data_2, axis=0)
    aggregated_data_3 = np.mean(data_3, axis=0)
    aggregated_data_4 = np.mean(data_4, axis=0)

else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Plot results for each volatility parameter
start_lag = 0
for param in params_volatility:
    N_lags = param['N_lags']
    end_lag = start_lag + N_lags

    def Psi(Hpsi):
        p = []
        window = param['window']
        N_lags = param['N_lags']
        factor = window**(2 * Hpsi)
        p.append(factor * (Phi_Hl(0, Hpsi) + 2 * Phi_Hl(1, Hpsi)))
        for i in range(2, N_lags + 1):
            p.append(factor * Phi_Hl(i, Hpsi))
        return np.array(p)


    # GMM Estimation on the entire dataset (QV_total)
    H_1, _ = estimation_GMM(np.identity(len(aggregated_data_1[start_lag:end_lag])),
                                    aggregated_data_1[start_lag:end_lag],
                                    Psi,
                                    0.001,
                                    0.499,
                                    0.0001)
    H_2, _ = estimation_GMM(np.identity(len(aggregated_data_1[start_lag:end_lag])),
                                    aggregated_data_2[start_lag:end_lag],
                                    Psi,
                                    0.001,
                                    0.499,
                                    0.0001)
    H_3, _ = estimation_GMM(np.identity(len(aggregated_data_1[start_lag:end_lag])),
                                    aggregated_data_3[start_lag:end_lag],
                                    Psi,
                                    0.001,
                                    0.499,
                                    0.0001)
    H_4, _ = estimation_GMM(np.identity(len(aggregated_data_1[start_lag:end_lag])),
                                    aggregated_data_4[start_lag:end_lag],
                                    Psi,
                                    0.001,
                                    0.499,
                                    0.0001)
    
    print(H_1)
    print(H_2)
    print(H_3)
    print(H_4)
    
    print()

    # Update starting lag for the next iteration
    start_lag = end_lag