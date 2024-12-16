import numpy as np
import matplotlib.pyplot as plt
import random
import time

from data_handler import DataHandler
from dates import FOMC_announcement, trading_halt
from price import Price
from volatility import VolatilityEstimator, volatility_pattern, bipower_average_V, Volatility, VolatilityPattern
from quadratic_variation import QuadraticCovariationsEstimator, AsymptoticVarianceEstimator
from estimator_H import estimation_01_2, Phi_Hl, estimation_GMM

def kernel_k(x):
    x = np.abs(x)
    if x <= 0.5:
        return 1-6*x**2 -6*x**3
    else:
        return 2*(1-x)**3

def Psi(H, params):
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
    for param in params:
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


if __name__ == "__main__":
    print("Initializing DataHandler...")

    # Initialize the data handler
    DH = DataHandler(
        prices_folder="~/Documents/data/SPY/price/1s/daily_csv/",
        tmp_folder="~/Documents/data/tmp/hurst_inference"
    )

    # Remove all FOMC announcement dates from the data
    print("Removing FOMC announcement dates...")
    for date in FOMC_announcement:
        DH.remove_date(date)

    # Remove all trading halt dates from the data
    print("Removing trading halt dates...")
    for date in trading_halt:
        DH.remove_date(date)

    # # Define parameters
    # asset = 'spy'
    # subsampling = 1
    # price_truncation_method = 'BIVAR3'
    # vol_truncation_method = 'STD3'
    # delta = 1.0 / (252.0 * 23400) * subsampling  # Time increment
    # Ln = 300
    # Kn = 300
    # W_fun = lambda Lmax, L: 1 

    # params = [
    #     {'window': 150, 'N_lags': 20},
    #     {'window': 300, 'N_lags': 5},
    #     {'window': 600, 'N_lags': 5},
    #     {'window': 1200, 'N_lags': 2},
    # ]




    # Define parameters
    asset = 'spy'
    subsampling = 5
    price_truncation_method = 'BIVAR3'
    vol_truncation_method = 'STD3'
    delta = 1.0 / (252.0 * 23400) * subsampling  # Time increment
    Ln = 600
    Kn = 300
    W_fun = lambda Lmax, L: kernel_k(L/Lmax) 

    params = [
        {'window': 60, 'N_lags': 8},
        {'window': 90, 'N_lags': 5},
        {'window': 120, 'N_lags': 3},
        {'window': 180, 'N_lags': 2},
    ]


    # Initialize the asymptotic variance estimator
    ave = AsymptoticVarianceEstimator(W_fun, Ln, Kn)

    # Get all dates
    all_price_files = [f for f in DH.price_files if f.startswith(asset+'_')]
    all_dates = [f.split('_')[1].replace('.csv','') for f in all_price_files]

    print("Precompute volatility patterns...")

    # Precompute volatility patterns
    window_array = []
    lags_array = []
    label_array = []

    for param in params:
        window = param['window']
        N_lags = param['N_lags']

        # Initialize volatility and quadratic variation estimators
        param["ve"] = VolatilityEstimator(
            delta=delta,
            window=window,
            price_truncation=price_truncation_method
        )
        param["qve"] = QuadraticCovariationsEstimator(
            window=window,
            N_lags=N_lags + 1,
            vol_truncation=vol_truncation_method
        )
        param["pattern"] = VolatilityPattern()

        # Extend window_array with repeated window entries
        window_array.extend([window for _ in range(N_lags)])
        lags_array.append("Lag_0 + 2 * Lag_1")
        lags_array.extend([f"Lag_{i}" for i in range(2,N_lags+1)])
        label_array.append(f"W{window}; L0 + 2 L1")
        label_array.extend([f"W{window}; L{i}" for i in range(2,N_lags+1)])


        print(f"Computing and accumulating patterns for window={window}, N_lags={N_lags}...")
        # Loop over all dates to compute and accumulate patterns
        for i, d in enumerate(all_dates):
            if i % 50 == 0:
                print(f"  Processing date {i+1}/{len(all_dates)}: {d}")

            y, m, day = map(int, d.split('-'))

            # Get price data
            price = DH.get_price(asset, y, m, day)
            price.subsample(subsampling)
            price_array = price.get_price()  # numpy array of prices

            # Compute volatility using VolatilityEstimator
            vol = param["ve"].compute(price_array)

            # Accumulate patterns
            param["pattern"].accumulate(vol)
        
        param["pattern"] = param["pattern"].get_pattern()

    print("Volatility patterns precomputed.")

    # Initialize accumulation arrays
    sum_V = np.zeros(len(window_array))
    sum_Sigma = np.zeros((len(window_array), len(window_array)))

    print("Computing sum_V and sum_Sigma across all dates...")
    start_time = time.time()

    # Loop again over all dates to compute DRV and update sum_V and sum_Sigma
    for i, d in enumerate(all_dates):
        if i % 50 == 0 and i > 0:
            elapsed_time = time.time() - start_time
            processed_percentage = (i + 1) / len(all_dates)
            estimated_total_time = elapsed_time / processed_percentage
            remaining_time = estimated_total_time - elapsed_time
            estimated_finish = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time + estimated_total_time))

            print(f"  Processing date {i+1}/{len(all_dates)}: {d}\tElapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s, Estimated Finish: {estimated_finish}")

        y, m, day = map(int, d.split('-'))

        # Get price data
        price = DH.get_price(asset, y, m, day)
        price.subsample(subsampling)
        price_array = price.get_price()

        # Compute daily Realized Variation (DRV) for each param set
        DRV = []
        for param in params:
            # Extract objects
            ve = param["ve"] 
            qve = param["qve"] 
            pattern = param["pattern"]

            # Compute volatility
            vol = ve.compute(price_array)
            # Compute DRV
            DRV.extend(qve.DRV(vol.get_values(), pattern.get_values()))

        # Average of the DRV values
        V = [np.mean(drv) for drv in DRV]
        sum_V += np.array(V)

        # Apply correction and compute sum_Sigma terms
        psi = [ave.correction(drv) for drv in DRV]
        for idx_i in range(len(window_array)):
            for idx_j in range(len(window_array)):
                sum_Sigma[idx_i, idx_j] += ave.compute(
                    psi[idx_i], psi[idx_j], 
                    window_array[idx_i], window_array[idx_j]
                )

    # Normalize by the number of patterns
    sum_V = sum_V / len(window_array)
    sum_Sigma = sum_Sigma / len(window_array)

    sum_V = sum_V / np.mean(sum_V)
    sum_Sigma = sum_Sigma / np.mean(sum_Sigma)

    # Make beatiful plot of correlation matrix sum_Sigma
    # Generate a correlation matrix from sum_Sigma
    # correlation_matrix = np.corrcoef(sum_Sigma)

    # Plot the covariance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(sum_Sigma, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Covariance Value')
    plt.title('Covariance Matrix from sum_Sigma')
    plt.xticks(ticks=range(len(sum_Sigma)), labels=label_array, rotation=45)
    plt.yticks(ticks=range(len(sum_Sigma)), labels=label_array)
    plt.tight_layout()
    plt.show()


    # Compute W_n, the inverse of Sigma
    W_n = np.linalg.inv(sum_Sigma)

    # Plot W_n, the inverse of Sigma
    plt.figure(figsize=(10, 8))
    plt.imshow(W_n, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Inverse Covariance Value')
    plt.title('Inverse Covariance Matrix (W_n) from sum_Sigma')
    plt.xticks(ticks=range(len(W_n)), labels=label_array, rotation=45)
    plt.yticks(ticks=range(len(W_n)), labels=label_array)
    plt.tight_layout()
    plt.show()


    H = estimation_GMM(np.identity(len(sum_V)),
                       sum_V,
                       lambda H : Psi(H, params),
                       0.001,
                       0.499,
                       0.001)

    print(f"Estimator of $H$ with Identity Weight Matrix:\n{H}")

    H_values, F_values, min_index = estimation_GMM(np.identity(len(sum_V)),
                                                   sum_V,
                                                   lambda H : Psi(H, params),
                                                    0.001,
                                                    0.499,
                                                    0.001,
                                                    True)

    # Plot the full pattern
    plt.figure(figsize=(10,6))
    plt.plot(H_values, F_values, color='black', linewidth=1, label='function F')
    plt.title("Function F with Identity Matrix Weight Matrix")    
    plt.xlabel("$H$")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()




    H = estimation_GMM(W_n,
                       sum_V,
                       lambda H : Psi(H, params),
                       0.001,
                       0.499,
                       0.001)

    print(f"Result with Custom Weight Matrix (W_n):\n{H}")

    H_values, F_values, min_index = estimation_GMM(W_n,
                                                   sum_V,
                                                   lambda H : Psi(H, params),
                                                   0.001,
                                                   0.499,
                                                   0.001,
                                                   True)

    # Plot the full pattern
    plt.figure(figsize=(10,6))
    plt.plot(H_values, F_values, color='black', linewidth=1, label='function F')
    plt.title("Function F with Custom Weight Matrix (W_n)")
    plt.xlabel("$H$")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    



