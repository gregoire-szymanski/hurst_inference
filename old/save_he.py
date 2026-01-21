import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np


##### Preliminary functions

def Phi_Hl(l: int, H: float) -> float:
    """
    Compute the value of $\\Phi^H_\\ell$ using a finite difference formula.

    This function evaluates a discrete approximation based on powers of absolute values,
    commonly used in fractional Brownian motion and related models.

    :param l: Index $\\ell$ in the formula (integer).
    :param H: Hurst exponent $H$, controlling the memory effect (float).
    :return: Computed value of $\\Phi^H_\\ell$.
    """
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    return numerator / denominator

##### GMM estimator

def F_estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H: list, normalisation: float = 1) -> float:
    """
    Compute the GMM objective function $F(H, R)$ for given parameters.
    
    This function minimizes:
    
    $$ F(H, R) = (V - P)^T W (V - P) $$
    
    where $P$ is computed based on $H$.

    :param W: Weight matrix (numpy array).
    :param V: Observation vector (numpy array).
    :param Psi_func: Function $\\Psi(H)$ providing model predictions.
    :param H: Scalar Hurst exponent wrapped in a list.
    :param normalisation: Normalization factor for the function value.
    :return: Evaluated objective function value.
    """

    H = H[0]
    V = np.atleast_2d(V).reshape(-1, 1)
    Psi = np.atleast_2d(Psi_func(H)).reshape(-1, 1)
        
    term0 = V.T @ W @ V
    term1 = (Psi.T @ W @ V) + (V.T @ W @ Psi)
    term2 = Psi.T @ W @ Psi
    
    term0 = term0[0, 0]
    term1 = term1[0, 0]
    term2 = term2[0, 0]
    
    R = term1 / term2 / 2
    
    return normalisation * (term0 - R * term1 + term2 * R * R)




def F_GMM_get_R(W: np.ndarray, V: np.ndarray, Psi_func, H: float) -> float:
    V = np.atleast_2d(V).reshape(-1, 1)
    Psi = np.atleast_2d(Psi_func(H)).reshape(-1, 1)
        
    term0 = V.T @ W @ V
    term1 = (Psi.T @ W @ V) + (V.T @ W @ Psi)
    term2 = Psi.T @ W @ Psi
    
    term0 = term0[0, 0]
    term1 = term1[0, 0]
    term2 = term2[0, 0]
    
    R = term1 / term2 / 2
    
    return R


def estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H_min: float = 0.001, H_max: float = 0.499, mesh: float = 0.001, debug: bool = False):
    """
    Perform Generalized Method of Moments (GMM) estimation for the Hurst exponent.
    
    This method finds $H$ that minimizes the GMM objective function over a predefined grid.
    
    :param W: Weight matrix (numpy array).
    :param V: Observation vector (numpy array).
    :param Psi_func: Function returning model predictions $\\Psi(H)$.
    :param H_min: Minimum value for H search grid.
    :param H_max: Maximum value for H search grid.
    :param mesh: Step size for grid search.
    :param debug: If True, return intermediate results.
    :return: Estimated Hurst exponent.
    """
    H_values = np.arange(H_min, H_max, mesh)
    F_values = [F_estimation_GMM(W, V, Psi_func, [H]) for H in H_values]
    min_index = np.argmin(F_values)
    
    if debug:
        R_values = [F_GMM_get_R(W, V, Psi_func, H) for H in H_values]
        return H_values, F_values, min_index, R_values

    return H_values[min_index], F_GMM_get_R(W, V, Psi_func, H_values[min_index])



# main parameters (defaults)

input_data = None

price_truncation_mode = None  # None, STD_X, BIVAR_X (X int/float)
volatility_truncation_mode = None  # None, STD_X (X int/float)
remove_pattern = None  # None, multiplicative, additive

volatility_window_size = None  # Integer

hurst_min_value = None  # Float
hurst_max_value = None  # Float
hurst_step = None  # Float

remove_FOMC_days = True  # True or False
remove_trading_halts = -1  # Integer, number of periods threshold, -1 to not remove
normalise_average_value = True  # True or False, default True

# Hurst-estimation specific parameters
initial_mesh = None  # Integer or float (seconds)
N_autocorrelation = None  # Integer (must be larger than 2)
compute_confidence_interval = False  # True or False, default is False
GMM_weight = "identity"  # "identity" or "optimal"
Ln = 180  # Integer, default value 180
Kn = 720  # Integer, default value 720
W_fun_id = "parzen"  # Only allowed value is 'parzen'


# Configuration file path

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "hurst_config.json")


# Helpers

def load_config(path: str = _CONFIG_FILE) -> None:
    """
    Load parameter values from the JSON config file into module globals.
    If the file does not exist, it is created with default values.
    """
    global input_data
    global normalise_average_value
    global price_truncation_mode, volatility_truncation_mode, remove_pattern
    global volatility_window_size, hurst_min_value, hurst_max_value, hurst_step
    global remove_FOMC_days, remove_trading_halts
    global initial_mesh, N_autocorrelation
    global compute_confidence_interval, GMM_weight, Ln, Kn, W_fun_id

    if not os.path.exists(path):
        save_config(path)
        return

    try:
        with open(path, "r") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        # On any read/parse error, keep current in-memory defaults.
        return

    input_data = config.get("input_data", input_data)
    price_truncation_mode = config.get("price_truncation_mode", price_truncation_mode)
    volatility_truncation_mode = config.get("volatility_truncation_mode", volatility_truncation_mode)
    remove_pattern = config.get("remove_pattern", remove_pattern)
    volatility_window_size = config.get("volatility_window_size", volatility_window_size)
    hurst_min_value = config.get("hurst_min_value", hurst_min_value)
    hurst_max_value = config.get("hurst_max_value", hurst_max_value)
    hurst_step = config.get("hurst_step", hurst_step)
    remove_FOMC_days = config.get("remove_FOMC_days", remove_FOMC_days)
    remove_trading_halts = config.get("remove_trading_halts", remove_trading_halts)
    normalise_average_value = config.get("normalise_average_value", normalise_average_value)

    initial_mesh = config.get("initial_mesh", initial_mesh)
    N_autocorrelation = config.get("N_autocorrelation", N_autocorrelation)
    compute_confidence_interval = config.get("compute_confidence_interval", compute_confidence_interval)
    GMM_weight = config.get("GMM_weight", GMM_weight)
    Ln = config.get("Ln", Ln)
    Kn = config.get("Kn", Kn)
    W_fun_id = config.get("W_fun_id", W_fun_id)


def save_config(path: str = _CONFIG_FILE) -> None:
    """
    Save current parameter values from module globals into the JSON config file.
    """
    config = {
        "input_data": input_data,
        "price_truncation_mode": price_truncation_mode,
        "volatility_truncation_mode": volatility_truncation_mode,
        "remove_pattern": remove_pattern,
        "volatility_window_size": volatility_window_size,
        "hurst_min_value": hurst_min_value,
        "hurst_max_value": hurst_max_value,
        "hurst_step": hurst_step,
        "remove_FOMC_days": remove_FOMC_days,
        "remove_trading_halts": remove_trading_halts,
        "normalise_average_value": normalise_average_value,
        "initial_mesh": initial_mesh,
        "N_autocorrelation": N_autocorrelation,
        "compute_confidence_interval": compute_confidence_interval,
        "GMM_weight": GMM_weight,
        "Ln": Ln,
        "Kn": Kn,
        "W_fun_id": W_fun_id,
    }

    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


# Helper functions

def bipower_average_V(price, window):
    n = len(price)
    if n <= 2 * window:  # Ensure there's enough data
        print("Not enough data points.")
        return -1.0

    # Compute price increments over the given window
    price_increments = price[window:] - price[:-window]

    # Calculate bipower average volatility
    sum_ = np.sum(np.abs(price_increments[window:] * price_increments[:-window]))

    # Calculate the final result
    mean = sum_ / (n - 2 * window)
    return (mean / window) * (np.pi / 2)


def truncate_absolute(values, threshold):
    values = np.asarray(values).copy()
    values[np.abs(values) > threshold] = 0
    return values


def compute_volatility_squared(price, window_size, truncation_method=None, truncation_param=None):
    price = np.asarray(price, dtype=float)
    if np.any(price <= 0):
        raise ValueError("Price contains non-positive values; cannot take log.")

    log_price = np.log(price)
    log_returns = log_price[1:] - log_price[:-1]

    n, N = len(log_returns), 0

    if truncation_method is not None:
        if truncation_method == 'STD':
            std_dev = np.std(log_returns)
            threshold = float(truncation_param) * std_dev
            N = np.sum(np.abs(log_returns) > threshold)
            log_returns = truncate_absolute(log_returns, threshold)
        elif truncation_method == 'BIVAR':
            bpa = bipower_average_V(log_price, window_size)
            if bpa <= 0:
                # fall back to no truncation if bpa invalid
                pass
            else:
                threshold = float(truncation_param) * np.sqrt(bpa)
                N = np.sum(np.abs(log_returns) > threshold)
                log_returns = truncate_absolute(log_returns, threshold)

    rv = np.concatenate([[0.0], np.cumsum(log_returns ** 2)])
    volatilities_squared = (rv[window_size:] - rv[:-window_size]) / float(window_size)
    return volatilities_squared, n, N

def compute_autocorrelation(vol_squared, N_lags, window, truncation_method=None, truncation_param=None):
    vol_squared = np.asarray(vol_squared, dtype=float)
    n = len(vol_squared)
    if n <= N_lags*window:
        raise ValueError("Not enough data points to compute autocorrelation with the given number of lags.")

    vol_squared_increments = vol_squared[window:] - vol_squared[:-window]

    if truncation_method is not None:
        if truncation_method == 'STD':
            std_dev = np.std(vol_squared_increments)
            threshold = float(truncation_param) * std_dev
            vol_squared_increments = truncate_absolute(vol_squared_increments, threshold)
        else:
            raise ValueError(f"Unknown truncation method: {truncation_method}")

    # mean_vol = np.mean(vol_squared_increments)
    mean_vol = 0
    autocorr = np.zeros(N_lags)

    for lag in range(N_lags):
        if lag == 0:
            autocorr[lag] = np.mean((vol_squared_increments - mean_vol) ** 2)
        else:
            autocorr[lag] = np.mean((vol_squared_increments[lag*window:] - mean_vol) * (vol_squared_increments[:-lag*window] - mean_vol))

    autocorr[1] = autocorr[0] + 2 * autocorr[1]

    return autocorr[1:]


def parse_truncation_mode(mode: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse truncation modes like:
      - None
      - "STD_3" or "STD_3.5"
      - "BIVAR_4" or "BIVAR_2.0"
    Returns (method, param) where method in {"STD","BIVAR"} or None.
    """
    if mode is None:
        return None, None
    if not isinstance(mode, str):
        raise ValueError(f"Invalid truncation mode type: {type(mode)}")
    m = re.match(r"^(STD|BIVAR)_(\d+(\.\d+)?)$", mode.strip().upper())
    if not m:
        raise ValueError(f"Invalid truncation mode format: {mode}. Expected None, STD_X or BIVAR_X.")
    method = m.group(1)
    param = float(m.group(2))
    return method, param



# Load FOMC dates list (YYYY-MM-DD per line)
with open(os.path.join(os.path.dirname(__file__), "dates", "FOMC.txt")) as _fomc_file:
    FOMC_dates = [line.strip() for line in _fomc_file if line.strip()]


# Load configuration on import so parameters are ready for use.
load_config()


def create_Psi_function(window: int, N_lags: int):
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

        factor = window**(2 * H)
        
        # Compute the first two terms outside the loop
        p.append(factor * (Phi_Hl(0, H) + 2 * Phi_Hl(1, H)))

        # Compute remaining terms for i in [2, N_lags]
        for i in range(2, N_lags):
            p.append(factor * Phi_Hl(i, H))

        return np.array(p)
    return Psi



def run_pipeline() -> None:
    # Step 0
    print("Step 0/7: Checking input/output folders and creating output folder if needed...")

    if input_data is None:
        raise ValueError("Config error: input_data is None.")

    if volatility_window_size is None or int(volatility_window_size) <= 0:
        raise ValueError("Config error: volatility_window_size must be a positive integer.")
    window = int(volatility_window_size)

    # Parse truncation
    price_trunc_method, price_trunc_param = parse_truncation_mode(price_truncation_mode)
    vol_trunc_method, vol_trunc_param = parse_truncation_mode(volatility_truncation_mode)

    # Step 1
    print("Step 1/7: Listing files, filtering by prefix+date format, loading prices, applying filters...")
    X = np.load(input_data, allow_pickle=True)

    n_day, price_per_day = X.shape
    daily_prices = [X[i, :] for i in range(n_day)]

    # Step 2
    print("Step 2/7: Computing daily volatility-squared series for each day...")

    daily_volatility_squared_list: List[np.ndarray] = []

    n_total = 0
    N_total = 0

    for prices in daily_prices:
        try:
            vsq, n, N = compute_volatility_squared(
                prices,
                window_size=window,
                truncation_method=price_trunc_method,
                truncation_param=price_trunc_param,
            )
            if vsq.size == 0 or not np.all(np.isfinite(vsq)):
                continue
            daily_volatility_squared_list.append(vsq.astype(float))
            n_total += n
            N_total += N
        except Exception:
            continue

    if not daily_volatility_squared_list:
        print("No volatility series could be computed.")
        return
    
    print(f"Total log-returns processed: n={n_total}, truncated points: N={N_total}, proportion: p={N_total / n_total if n_total > 0 else 0.0:.6f}")

    # Step 3
    print("Step 3/7: Normalising and removing intraday volatility pattern if applicable...")

    # Align lengths (pattern removal and averaging require same intraday index)
    min_len = min(v.shape[0] for v in daily_volatility_squared_list)
    daily_volatility_squared_list = [v[:min_len].copy() for v in daily_volatility_squared_list]

    # Normalise average value per day
    if normalise_average_value:
        for i, v in enumerate(daily_volatility_squared_list):
            avg = float(np.mean(v))
            if avg != 0.0 and np.isfinite(avg):
                daily_volatility_squared_list[i] = v / avg


    # Remove intraday pattern
    if remove_pattern is not None:
        rp = str(remove_pattern).strip().lower()
        if rp == "multiplicative":
            stacked = np.vstack(daily_volatility_squared_list)  # shape (n_days, min_len)
            pattern = np.mean(stacked, axis=0)
            # avoid divide-by-zero
            pattern = np.where(pattern == 0.0, 1.0, pattern)
            for i, v in enumerate(daily_volatility_squared_list):
                daily_volatility_squared_list[i] = v / pattern
        elif rp == "additive":
            raise ValueError("remove_pattern='additive' is not supported.")
        else:
            raise ValueError(f"Invalid remove_pattern: {remove_pattern}. Expected None, 'multiplicative', or 'additive'.")
    

    # Step 4: Estimate autocorrelation vectors
    print("Step 4/7: Estimating autocorrelation vectors...")

    if N_autocorrelation is None or int(N_autocorrelation) <= 2:
        raise ValueError("Config error: N_autocorrelation must be an integer greater than 2.")
    n_lags = int(N_autocorrelation)

    series_len = daily_volatility_squared_list[0].shape[0]
    if series_len <= n_lags:
        raise ValueError(
            f"Config error: N_autocorrelation ({n_lags}) must be smaller than the volatility series length ({series_len})."
        )

    daily_autocorr_vectors: List[np.ndarray] = []
    for vsq in daily_volatility_squared_list:
        daily_autocorr_vectors.append(compute_autocorrelation(vsq, n_lags, window, truncation_method=vol_trunc_method, truncation_param=vol_trunc_param))

    daily_autocorr_matrix = np.vstack(daily_autocorr_vectors)
    average_autocorrelation = np.mean(daily_autocorr_matrix, axis=0)

    # print(f"Average autocorrelation vector: {average_autocorrelation}")

    # Step 5: Estimate covariance matrices
    print("Step 5/7: Estimating covariance matrices...")
    if GMM_weight not in {"identity", "optimal"}:
        raise ValueError(f"Invalid GMM_weight: {GMM_weight}. Expected 'identity' or 'optimal'.")
    if GMM_weight == "optimal" or compute_confidence_interval:
        raise NotImplementedError("Covariance matrix computation not yet implemented.")
    else:
        print("Skipping covariance matrix computation (GMM_weight='identity' and no confidence interval).")

    # Step 6: GMM estimation of Hurst exponent
    print(
        f"Step 6/7: Estimating Hurst exponent via GMM on grid H in "
        f"[{hurst_min_value}, {hurst_max_value}] with step {hurst_step}..."
    )

    Psi = create_Psi_function(window, n_lags)

    H_total, _ = estimation_GMM(np.identity(len(average_autocorrelation)),
                            average_autocorrelation,
                            Psi,
                            hurst_min_value,
                            hurst_max_value,
                            hurst_step)
    
    # Step 7: Save results to output folder and print results as well
    print("Step 7/7: Saving results and printing summary...")
    print(f"Estimated Hurst exponent: {H_total}")

if __name__ == "__main__":
    run_pipeline()
