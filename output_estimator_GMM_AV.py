import numpy as np
import matplotlib.pyplot as plt
import random

from data_handler import DataHandler
from dates import FOMC_announcement, trading_halt
from price import Price
from volatility import VolatilityEstimator, volatility_pattern, bipower_average_V, Volatility, VolatilityPattern
from quadratic_variation import QuadraticCovariationsEstimator, AsymptoticVarianceEstimator
from estimator_H import estimation_01_2, Phi_Hl, estimation_GMM


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

    # Define parameters
    asset = 'spy'
    subsampling = 1
    price_truncation_method = 'BIVAR3'
    vol_truncation_method = 'STD3'
    delta = 1.0 / (252.0 * 23400) * subsampling  # Time increment
    Ln = 300
    Kn = 300
    W_fun = lambda Lmax, L: 1 

    params = [
        {'window': 150, 'N_lags': 20},
        {'window': 300, 'N_lags': 10},
        {'window': 600, 'N_lags': 5},
        {'window': 1200, 'N_lags': 2},
    ]

    # Initialize the asymptotic variance estimator
    ave = AsymptoticVarianceEstimator(W_fun, Ln, Kn)

    # Get all dates
    all_price_files = [f for f in DH.price_files if f.startswith(asset+'_')]
    all_dates = [f.split('_')[1].replace('.csv','') for f in all_price_files]

    print("Precompute volatility patterns...")

    # Precompute volatility patterns
    window_array = []
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
    # Loop again over all dates to compute DRV and update sum_V and sum_Sigma
    for i, d in enumerate(all_dates):
        if i % 50 == 0:
            print(f"  Processing date {i+1}/{len(all_dates)}: {d}")

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

    print("Computation completed.")
    print("sum_V:", sum_V)
    print("sum_Sigma:", sum_Sigma)



