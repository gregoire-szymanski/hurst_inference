import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from estimator_H import *
import os


params_volatility = [
    {'window': 120, 'N_lags': 6},
    {'window': 150, 'N_lags': 4},
]

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

total_n_lags = sum([param['N_lags'] for param in params_volatility])

# Folder paths and Hurst values
folder = "/Users/gregoire.szymanski/Documents/mc_results"
list_H_values = [0.1, 0.2, 0.3, 0.4, 0.5]
identificator = "5s"

for H in list_H_values:
    QV_list = []

        
    filename = f"results{int(H * 10):02d}_{identificator}.txt"
    file_path = os.path.join(folder, filename)
        
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = [np.array([float(value) for value in line.split(",")]) for line in lines]
        QV = [line[21:21+total_n_lags] for line in data]
        QV_total = np.sum(QV, axis=0) if QV else None
        H_total_id = estimation_GMM(np.identity(len(QV_total)),
                                    QV_total,
                                    Psi,
                                    0.001,
                                    0.499,
                                    0.00001)
        print(f"{H:.2}\t{H_total_id:.4}")
