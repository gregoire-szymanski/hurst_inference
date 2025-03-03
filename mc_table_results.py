import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lib.estimator_H import *
import os


params_volatility = [
    {'window': 60, 'N_lags': 12},
    {'window': 90, 'N_lags': 9},
    {'window': 120, 'N_lags': 6},
    {'window': 150, 'N_lags': 4},
]

subparam_params_volatility = [
    {'window': 120, 'N_lags': 6},
    {'window': 150, 'N_lags': 4},
]

# Folder paths and Hurst values
folder = "/Users/gregoire.szymanski/Documents/mc_results"
list_H_values = [0.1, 0.2, 0.3, 0.4, 0.5]    
identificator = '5s'
N_days = 252 * 4

# Optimisation parameters
H_mesh = 0.0001
H_min = H_mesh
H_max = 1 


def Psi(H):
    """
    Precompute the Psi(H) function for the given parameter configurations.
    Psi(H) uses the pre-defined parameters (window sizes and number of lags)
    to generate a set of values that depend on H.

    Parameters
    ----------
    H : float
        The Hurst exponent value to use for computations.
    params : list of dict
        A list of parameter configurations. Each dict contains:
        - 'window': int
        - 'N_lags': int

    Returns
    -------
    np.array
        A NumPy array of computed Psi values.
    """
    p = []
    for param in subparam_params_volatility:
        window = param['window']
        N_lags = param['N_lags']

        # factor = window^(2H)
        factor = window**(2 * H)
        
        # Compute the first two terms outside the loop
        p.append(factor * (Phi_Hl(0, H) + 2 * Phi_Hl(1, H)))

        # Compute remaining terms for i in [2, N_lags]
        for i in range(2, N_lags + 1):
            p.append(factor * Phi_Hl(i, H))
    return np.array(p)


# Params array

window_array = []
lags_array = []
label_array = []

total_n_lags = sum([param['N_lags'] for param in params_volatility])


subparam_mask = []
for param in params_volatility:
    if param in subparam_params_volatility:
        subparam_mask.extend([True] * param['N_lags'])
    else:
        subparam_mask.extend([False] * param['N_lags'])

# Data storage for final table
table_data = []


for H in list_H_values:
    QV_list = []
        
    filename = f"results{int(H * 10):02d}_{identificator}.txt"
    file_path = os.path.join(folder, filename)
        
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = [np.array([float(value) for value in line.split(",")]) for line in lines]
        
        QV = [line[:total_n_lags] for line in data]
        QV = [line[subparam_mask] for line in QV]
        
        N_estimation_results = len(QV) // N_days
        results = []

        for i in range(N_estimation_results):
        
            QV_total = np.sum(QV[i*N_days: (i+1)*N_days], axis=0) if QV else None
            QV_total[0] = QV_total[0] 
            QV_total[6] = QV_total[6]

            H_total_id, _ = estimation_GMM(np.identity(len(QV_total)),
                                        QV_total,
                                        Psi,
                                        H_min,
                                        H_max,
                                        H_mesh)
            print(f"{H:.2}\t{H_total_id:.4}")

            results.append(H_total_id)

        print()

        # Compute summary statistics
        summary = {
            "H Value": H,
            "Mean": np.mean(results),
            "Median": np.median(results),
            "Standard Deviation": np.std(results),
            "Minimum": np.min(results),
            "5th Percentile": np.percentile(results, 5),
            "25th Percentile (Q1)": np.percentile(results, 25),
            "75th Percentile (Q3)": np.percentile(results, 75),
            "95th Percentile": np.percentile(results, 95),
            "Maximum": np.max(results),
        }
        table_data.append(summary)


# Convert results to a DataFrame and display as a table
df_results = pd.DataFrame(table_data)
print(df_results.to_string(index=False, float_format="{:.4f}".format))