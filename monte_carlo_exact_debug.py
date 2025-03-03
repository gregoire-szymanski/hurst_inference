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
window = 60
H = 0.1

# Calculate theoretical limits
phi_values = [Phi_Hl(l, H) for l in range(13)]
phi_values[1] = phi_values[0] + 2*phi_values[1] 
theoretical_limit = np.array(phi_values)[1:]



param = params_volatility[0]
start_lag = 0
end_lag = 12
N_lags = 12
data = np.array([ 1.,-0.17572735,-0.06615738,-0.03660852,-0.02492819,-0.01791,-0.01316616,-0.01057582,-0.00918582,-0.00704541,-0.00612518,-0.00463688]) * (window)**(2*H)

print(data)

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

H_values = np.arange(.01, 0.2, 0.01)
F_values = [F_estimation_GMM(identity, data, Psi, [H]) for H in H_values]
min_index = np.argmin(F_values)

plt.plot(H_values, F_values)
plt.show()