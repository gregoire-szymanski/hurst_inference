import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from estimator_H import *
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

subparam_mask = []
for param in params_volatility:
    if param in subparam_params_volatility:
        subparam_mask.extend([True] * param['N_lags'])
    else:
        subparam_mask.extend([False] * param['N_lags'])
    
print(subparam_mask)

# Optimisation parameters
H_min = 0
H_max = 0.5
H_step = 1000
H_mesh = (H_max - H_min) / H_step

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
    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']

        # factor = window^(2H)
        factor = window**(2 * H)
        
        # Compute the first two terms outside the loop
        p.append(factor * (Phi_Hl(0, H) + Phi_Hl(1, H)))

        # Compute remaining terms for i in [2, N_lags]
        for i in range(2, N_lags + 1):
            p.append(factor * Phi_Hl(i, H))
    return np.array(p)

# Params array

window_array = []
lags_array = []
label_array = []

for param in params_volatility:
    window = param['window']
    N_lags = param['N_lags']
    
    # Extend window_array with repeated window entries
    window_array.extend([window for _ in range(N_lags)])
    lags_array.append("Lag_0 + 2 * Lag_1")
    lags_array.extend([f"Lag_{i}" for i in range(2,N_lags+1)])
    label_array.append(f"W{window}; L0 + 2 L1")
    label_array.extend([f"W{window}; L{i}" for i in range(2,N_lags+1)])

total_n_lags = sum([param['N_lags'] for param in params_volatility])

# Folder paths and Hurst values
folder = "/Users/gregoire.szymanski/Documents/mc_raw_results"
identificator = '5s'
list_sub_folders = os.listdir(folder)
list_H_values = [0.1, 0.2, 0.3, 0.4, 0.5]

for H in list_H_values:
    QV_list = []

    for sub in list_sub_folders:        
        sub_path = os.path.join(folder, sub)
        filename = f"results{int(H * 10):02d}_{identificator}.txt"
        file_path = os.path.join(sub_path, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Parse data
            data = [np.array([float(value) for value in line.split(",")]) for line in lines]
            QV = [line[:total_n_lags] for line in data]
            # AV = data[total_n_lags:total_n_lags + total_n_lags**2].reshape((total_n_lags, total_n_lags))

            # Aggregate data
            QV_total = np.sum(QV, axis=0) if QV else None
            # AV_total = np.sum(list_lines_AV, axis=0) if list_lines_AV else None

            # list_lines_AV.append(AV)

            # GMM Estimation on the entire dataset (QV_total)
            H_total_id = estimation_GMM(np.identity(len(QV_total)),
                                        QV_total,
                                        Psi,
                                        0.001,
                                        0.499,
                                        0.00001)
            
            # for lab, qv in zip(label_array, QV_total):
                # print(f"{lab}\t{qv*10**5:.3}")
            
            QV_list.append(QV_total)
            print(f"{H:.2}\t{H_total_id:.4}")
            # print()
    print()
    QV_total = np.sum(QV_list, axis=0) if QV_list else None
    H_total_id = estimation_GMM(np.identity(len(QV_total)),
                                        QV_total,
                                        Psi,
                                        0.001,
                                        0.499,
                                        0.00001)
    print(f"{H:.2}\t{H_total_id:.4}")
    print()

