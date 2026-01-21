import matplotlib.pyplot as plt
import numpy as np

from lib.volatility import * 
from lib.estimator_H import * 

covariances = []
window = 300
max_lag = 5

for i in range(10):
    # Parameters
    bin_file = f'/Users/gregoireszymanski/Documents/data/rough_heston/simulation_010/simulation_{i+1:02d}.bin'

    nb_days = 1000
    T = 1.0 / 252.0
    dt = 1.0 / 60.0 / 60.0 / 8.0 / 252.0
    n = int(T/dt)


    # Read the binary file as float64
    data = np.fromfile(bin_file, dtype=np.float64)
    expected = nb_days * (n + 1)
    if data.size != expected:
        raise ValueError(f"Expected {expected} doubles, but got {data.size}")

    # Reshape into a 3D array: (simulations, days, timesteps+1)
    vol2d = data.reshape((nb_days, n + 1))

    

    for i in range(nb_days):
        rv = np.cumsum(vol2d[i])
        rv = (rv[window:] - rv[:-window]) / window
        inc = rv[window:] - rv[:-window]
        local_covariance = []
        #local_covariance.append(np.mean(inc**2) + 2 * np.mean(inc[window:]*inc[:-window]))
        #for i in range(2, max_lag + 1):
        #    local_covariance.append(np.mean(inc[i * window:]*inc[:-i * window]))
        local_covariance.append(np.mean(inc**2))
        for i in range(1, max_lag + 1):
            local_covariance.append(np.mean(inc[i * window:]*inc[:-i * window]))
        covariances.append(local_covariance)

covariances = np.array(covariances)
covariances = np.mean(covariances, axis=0)



params_volatility = [
    {'window': window, 'N_lags': max_lag}
#   {'window': 150, 'N_lags': 4}
]


# Create estimation functions
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
        p.append(factor * (Phi_Hl(0, H)))

        # Compute remaining terms for i in [2, N_lags]
        for i in range(1, N_lags + 1):
            p.append(factor * Phi_Hl(i, H))
    return np.array(p)



H_mesh = 0.001
H_min = H_mesh
H_max = 0.5 + H_mesh



# GMM Estimation on the entire dataset (QV_total)
H_total, _ = estimation_GMM(np.identity(len(covariances)),
                            covariances,
                            Psi,
                            H_min,
                            H_max,
                            H_mesh)
print(H_total)

real_covariances = [Phi_Hl(i, 0.1) for i in range(max_lag+1)]
fit_covariances = [Phi_Hl(i, H_total) for i in range(max_lag+1)]

lags = np.arange(len(covariances))

# Use a clean style
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(8, 5))

plt.plot(lags, covariances / covariances[0],
         marker='o', linestyle='-', linewidth=2, markersize=6,
         label='Empirical')

plt.plot(lags, real_covariances / real_covariances[0],
         marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Theoretical (H=0.1)')

plt.plot(lags, fit_covariances / fit_covariances[0],
         marker='^', linestyle='-.', linewidth=2, markersize=6,
         label=f'Fitted (H={H_total:.3f})')

plt.xlabel('Lag')
plt.ylabel('Normalized Covariance')
plt.title('Normalized Lag‐Covariances: Empirical vs. Theoretical vs. Fitted')
plt.legend(loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()