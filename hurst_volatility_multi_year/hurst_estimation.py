import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize
from scipy.stats import norm

import numpy as np



print("Import done")

# main parameters (defaults)

input_data_folder = "new_data"

price_truncation_mode = 'BIVAR_3'  # None, STD_X, BIVAR_X (X int/float)
volatility_truncation_mode = 'STD_3'  # None, STD_X (X int/float)
remove_pattern = 'multiplicative'  # None, multiplicative, additive

volatility_window_size = 48 * 2  # Integer
increment_volatility_size = 60 * 2 # Integer

hurst_min_value = 0.0001  # Float
hurst_max_value = 0.4999  # Float
hurst_step = 0.0001  # Float

normalise_average_value = True  # True or False, default True

N_autocorrelation = 6  # Integer (must be larger than 2)
compute_confidence_interval = False  # True or False, default is False
GMM_weight = "identity"  # "identity" or "optimal"
Ln = 180  # Integer, default value 180
Kn = 720  # Integer, default value 720
W_fun_id = "parzen"  # Only allowed value is 'parzen'
print_truncated_infos = False  # True or False, default is False
optimization_method = "BFGS"  # "brute" or "BFGS"

start_year = 2018  # None or Integer
end_year = 2022  # None or Integer
N_consecutive_years = None  # None or Integer






##### Preliminary functions

def Phi_Hl(l: int, H: float, kappa: float = 1) -> float:
    """
    Compute the value of $\\Phi^H_\\ell$ using a finite difference formula.

    This function evaluates a discrete approximation based on powers of absolute values,
    commonly used in fractional Brownian motion and related models.

    :param l: Index $\\ell$ in the formula (integer).
    :param H: Hurst exponent $H$, controlling the memory effect (float).
    :param kappa: Sampling ratio parameter used in the generalized stencil.
    :return: Computed value of $\\Phi^H_\\ell$.
    """
    h2 = 2.0 * H
    exponent = h2 + 2.0
    denominator = 2.0 * (h2 + 1.0) * (h2 + 2.0)

    # Fast path for the dominant case used by the rest of the module.
    if kappa == 1:
        return (
            abs(l + 2) ** exponent
            - 4.0 * abs(l + 1) ** exponent
            + 6.0 * abs(l) ** exponent
            - 4.0 * abs(l - 1) ** exponent
            + abs(l - 2) ** exponent
        ) / denominator

    inv_kappa = 1.0 / kappa
    base = abs(l) ** exponent
    numerator1 = (
        abs(l + 1 + inv_kappa) ** exponent
        + abs(l + 1 - inv_kappa) ** exponent
        + abs(l - 1 + inv_kappa) ** exponent
        + abs(l - 1 - inv_kappa) ** exponent
    )
    numerator2 = (
        abs(l + 1) ** exponent
        + abs(l + inv_kappa) ** exponent
        + abs(l - 1) ** exponent
        + abs(l - inv_kappa) ** exponent
    )
    return (kappa * kappa) * (numerator1 - 2.0 * numerator2 + 4.0 * base) / denominator


def dPhi_Hl_dH(l: int, H: float, kappa: float = 1) -> float:
    """
    Compute the derivative of $\\Phi^H_\\ell$ with respect to $H$.

    Uses the chain rule to differentiate power terms in the finite difference formula.

    :param l: Index $\\ell$ in the formula (integer).
    :param H: Hurst exponent $H$ (float).
    :param kappa: Sampling ratio parameter used in the generalized stencil.
    :return: The computed derivative $\\frac{d}{dH} \\Phi^H_\\ell$.
    """
    exponent = 2.0 * H + 2.0
    denominator = 2.0 * (2.0 * H + 1.0) * (2.0 * H + 2.0)
    denominator_derivative = 4.0 * (4.0 * H + 3.0)

    def pow_and_derivative(x: float) -> Tuple[float, float]:
        ax = abs(x)
        if ax == 0.0:
            return 0.0, 0.0
        value = ax ** exponent
        return value, 2.0 * value * np.log(ax)

    if kappa == 1:
        p_l2, dp_l2 = pow_and_derivative(l + 2)
        p_l1, dp_l1 = pow_and_derivative(l + 1)
        p_l0, dp_l0 = pow_and_derivative(l)
        p_lm1, dp_lm1 = pow_and_derivative(l - 1)
        p_lm2, dp_lm2 = pow_and_derivative(l - 2)

        numerator = p_l2 - 4.0 * p_l1 + 6.0 * p_l0 - 4.0 * p_lm1 + p_lm2
        numerator_derivative = dp_l2 - 4.0 * dp_l1 + 6.0 * dp_l0 - 4.0 * dp_lm1 + dp_lm2
    else:
        inv_kappa = 1.0 / kappa

        p1, dp1 = pow_and_derivative(l + 1 + inv_kappa)
        p2, dp2 = pow_and_derivative(l + 1 - inv_kappa)
        p3, dp3 = pow_and_derivative(l - 1 + inv_kappa)
        p4, dp4 = pow_and_derivative(l - 1 - inv_kappa)
        q1, dq1 = pow_and_derivative(l + 1)
        q2, dq2 = pow_and_derivative(l + inv_kappa)
        q3, dq3 = pow_and_derivative(l - 1)
        q4, dq4 = pow_and_derivative(l - inv_kappa)
        b, db = pow_and_derivative(l)

        kappa_sq = kappa * kappa
        numerator = kappa_sq * ((p1 + p2 + p3 + p4) - 2.0 * (q1 + q2 + q3 + q4) + 4.0 * b)
        numerator_derivative = kappa_sq * ((dp1 + dp2 + dp3 + dp4) - 2.0 * (dq1 + dq2 + dq3 + dq4) + 4.0 * db)

    return (numerator_derivative * denominator - numerator * denominator_derivative) / (denominator * denominator)

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

def estimation_GMM_brute(
    W: np.ndarray,
    V: np.ndarray,
    Psi_func,
    H_min: float = 0.001,
    H_max: float = 0.499,
    mesh: float = 0.001,
    debug: bool = False,
):
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


def estimation_GMM_BFGS(
    W: np.ndarray,
    V: np.ndarray,
    Psi_func,
    H_min: float = 0.001,
    H_max: float = 0.499,
    mesh: float = 0.001,
    debug: bool = False,
):
    """
    Perform 1D GMM estimation for the Hurst exponent using BFGS.

    The optimized objective remains F_estimation_GMM(W, V, Psi_func, [H]).
    """
    del mesh  # mesh is unused for BFGS, kept for API compatibility

    if H_min >= H_max:
        raise ValueError("H_min must be strictly smaller than H_max.")

    def objective(x: np.ndarray) -> float:
        H = float(np.atleast_1d(x)[0])
        return float(F_estimation_GMM(W, V, Psi_func, [H]))

    x0 = np.array([(H_min + H_max) / 2.0], dtype=float)
    result = minimize(objective, x0=x0, method="BFGS")

    H_estimated = float(np.clip(result.x[0], H_min, H_max))
    R_estimated = F_GMM_get_R(W, V, Psi_func, H_estimated)

    if debug:
        return H_estimated, R_estimated, result
    return H_estimated, R_estimated


def estimation_GMM(
    W: np.ndarray,
    V: np.ndarray,
    Psi_func,
    H_min: float = 0.001,
    H_max: float = 0.499,
    mesh: float = 0.001,
    debug: bool = False,
    optimization_method: str = "brute",
):
    method = str(optimization_method).strip().lower()
    if method == "brute":
        return estimation_GMM_brute(W, V, Psi_func, H_min, H_max, mesh, debug)
    if method == "bfgs":
        return estimation_GMM_BFGS(W, V, Psi_func, H_min, H_max, mesh, debug)
    raise ValueError("optimization_method must be either 'brute' or 'BFGS'.")

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

def compute_autocorrelation(
    vol_squared,
    N_lags,
    window,
    truncation_method=None,
    truncation_param=None,
    return_counts=False,
):
    vol_squared = np.asarray(vol_squared, dtype=float)
    n = len(vol_squared)
    if n <= N_lags*window:
        raise ValueError("Not enough data points to compute autocorrelation with the given number of lags.")

    vol_squared_increments = vol_squared[window:] - vol_squared[:-window]
    n_increments = len(vol_squared_increments)
    truncated = 0

    if truncation_method is not None:
        if truncation_method == 'STD':
            std_dev = np.std(vol_squared_increments)
            threshold = float(truncation_param) * std_dev
            truncated = int(np.sum(np.abs(vol_squared_increments) > threshold))
            vol_squared_increments = truncate_absolute(vol_squared_increments, threshold)
        else:
            raise ValueError(f"Unknown truncation method: {truncation_method}")

    # mean_vol = np.mean(vol_squared_increments)
    mean_vol = 0
    autocorr = np.zeros(N_lags)

    for lag in range(N_lags):
        if lag == 0:
            autocorr[lag] = np.mean((vol_squared_increments - mean_vol) ** 2) * n_increments / (n_increments - truncated)
        else:
            autocorr[lag] = np.mean((vol_squared_increments[lag*window:] - mean_vol) * (vol_squared_increments[:-lag*window] - mean_vol)) * n_increments / (n_increments - 2*truncated)

    autocorr[1] = autocorr[0] + 2 * autocorr[1]

    if return_counts:
        return autocorr[1:], n_increments, truncated
    return autocorr[1:]

def correct_DRV(DRV, Kn):
    DRV = np.asarray(DRV)
    if len(DRV) < Kn:
        raise ValueError("DRV length must be at least Kn.")
    # Simple moving average
    kernel = np.ones(Kn) / Kn
    psi = np.convolve(DRV, kernel, mode='valid')
    return psi

def compute_term(psi, psi_prime, kn, kn_prime, L):
    """
    Compute a single term in the asymptotic variance estimator for a given lag L.
    
    Parameters
    ----------
    psi : np.ndarray
        A time series of corrected values.
    psi_prime : np.ndarray
        Another time series of corrected values.
    kn : float
        A scaling parameter.
    kn_prime : float
        Another scaling parameter.
    L : int
        The lag at which to compute the term.
    
    Returns
    -------
    float
        The computed term.
    """
    psi = np.asarray(psi)
    psi_prime = np.asarray(psi_prime)
    
    # Align psi and psi_prime based on lag L
    if L > 0:
        # psi_prime is shifted forward by L relative to psi
        # so we must drop the last L elements of psi and the first L elements of psi_prime
        psi_trunc = psi[:-L]
        psi_prime_trunc = psi_prime[L:]
    elif L < 0:
        # psi is shifted forward by -L relative to psi_prime
        # so we must drop the first -L elements of psi and the last -L elements of psi_prime
        shift = -L
        psi_trunc = psi[shift:]
        psi_prime_trunc = psi_prime[:-shift]
    else:
        # L = 0, no shift
        psi_trunc = psi
        psi_prime_trunc = psi_prime
    
    # Ensure equal lengths
    N = min(len(psi_trunc), len(psi_prime_trunc))
    if N == 0:
        return 0.0
    
    psi_trunc = psi_trunc[:N]
    psi_prime_trunc = psi_prime_trunc[:N]
    
    # Compute mean of product
    val = np.sum(psi_trunc * psi_prime_trunc)
    return val / (kn * kn_prime)

def compute_covariance(W_fun, 
                       Ln,
                       Kn, 
                       vol_squared, 
                       N_lags, 
                       window, 
                       truncation_method=None, 
                       truncation_param=None):
    if N_lags < 2:
        raise ValueError("N_lags must be >= 2 so autocovariance has size N_lags-1.")

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
    
    DRV = []
    for lag in range(N_lags):
        if lag == 0:
            DRV.append(vol_squared_increments**2)
        else:
            DRV.append(vol_squared_increments[(lag * window):] * vol_squared_increments[: - (lag * window)])
    DRV[0] = DRV[0][:len(DRV[1])]
    DRV[1] = DRV[0] + 2 * DRV[1]
    DRV = DRV[1:]

    psi = [correct_DRV(drv, Kn) for drv in DRV]

    sigma = np.zeros((N_lags-1, N_lags-1))

    for idx_i in range(N_lags-1):
        for idx_j in range(N_lags-1):
            for L in range(1, Ln + 1):
                w = W_fun(Ln,L)
                term = compute_term(psi[idx_i], psi[idx_j], window, window, L)
                sigma[idx_i, idx_j] += w * term


    sigma = sigma + sigma.transpose()

    # for idx_i in range(N_lags-1):
    #     for idx_j in range(N_lags-1):
    #         w = W_fun(Ln,0)
    #         sigma[idx_i, idx_j] += compute_term(psi[idx_i], psi[idx_j], window, window, 0)

    return sigma

### Confidence interval ###

def uncorrected_alpha(theta, lag, H, kappa):
    return theta**(2*H-1) * dPhi_Hl_dH(lag, H, kappa) + 2 * np.log(theta) * Phi_Hl(lag, H, kappa)

def uncorrected_beta(theta, lag, H, kappa):
    return theta**(2*H-1) * Phi_Hl(lag, H, kappa)

def compute_alpha(theta, lag, H, kappa):
    if lag == 1:
        return uncorrected_alpha(theta, 0, H, kappa) + 2 * uncorrected_alpha(theta, 1, H, kappa)
    return uncorrected_alpha(theta, lag, H, kappa)


def compute_beta(theta, lag, H, kappa):
    if lag == 1:
        return uncorrected_beta(theta, 0, H, kappa) + 2 * uncorrected_beta(theta, 1, H, kappa)
    return uncorrected_beta(theta, lag, H, kappa)




def get_confidence_size(N_lags, window, kappa, H_estimated, R_estimated, n_days, delta_n, Sigma_estimated, W_chosen):
    theta = 1

    alpha = np.zeros(N_lags-1)
    beta = np.zeros(N_lags-1)

    for i in range(1,N_lags):
        alpha[i-1] = compute_alpha(theta, i, H_estimated, kappa)
        beta[i-1] = compute_beta(theta, i, H_estimated, kappa)

    u_t = np.array([alpha, beta]).transpose()

    D = np.array([
        [1, 0],
        [-2 * np.log(window * delta_n) * R_estimated, 1]
    ])

    uWu_inv = np.linalg.inv(u_t.transpose() @ W_chosen @ u_t)
    matrix_43 = (delta_n * window)**(1-4*H_estimated) * window * delta_n * D @ uWu_inv @ u_t.transpose() @ W_chosen @ Sigma_estimated @ W_chosen @ u_t @ uWu_inv @ D.transpose()

    return matrix_43[0,0]**0.5, matrix_43[1,1]**0.5
    # return matrix_43[0,0]**0.5 / np.sqrt(n_days), matrix_43[1,1]**0.5 / np.sqrt(n_days)



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


def create_Psi_function(window: int, N_lags: int, kappa: float):
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
        p.append(factor * (Phi_Hl(0, H, kappa) + 2 * Phi_Hl(1, H, kappa)))

        # Compute remaining terms for i in [2, N_lags]
        for i in range(2, N_lags):
            p.append(factor * Phi_Hl(i, H, kappa))

        return np.array(p)
    return Psi



def _run_pipeline_from_array(
    X: np.ndarray,
    price_truncation_mode: Optional[str] = None,
    volatility_truncation_mode: Optional[str] = None,
    remove_pattern: Optional[str] = None,
    volatility_window_size: Optional[int] = None,
    increment_volatility_size: Optional[int] = None,
    hurst_min_value: Optional[float] = None,
    hurst_max_value: Optional[float] = None,
    hurst_step: Optional[float] = None,
    delta_n: Optional[float] = None,
    normalise_average_value: bool = True,
    N_autocorrelation: Optional[int] = None,
    compute_confidence_interval: bool = False,
    GMM_weight: str = "identity",
    Ln: int = 180,
    Kn: int = 720,
    W_fun_id: str = "parzen",
    print_truncated_infos: bool = False,
    optimization_method: str = "brute",
) -> Optional[Tuple[float, Optional[float]]]:
    """Run the end-to-end Hurst inference pipeline on a price array."""
    # Step 0
    # print("Step 0/7: Checking input/output folders and creating output folder if needed...")


    W_fun = None
    if W_fun_id == "parzen":
        kernel_k = lambda x: 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x)**3
        W_fun = lambda Lmax, L: kernel_k(np.abs(L / Lmax))
    else:
        raise ValueError(f"Invalid W_fun_id: {W_fun_id}. Expected 'parzen'.")
    
    if volatility_window_size is None or int(volatility_window_size) <= 0:
        raise ValueError("Config error: volatility_window_size must be a positive integer.")
    window = int(volatility_window_size)

    if increment_volatility_size is None:
        increment_window = window
    else:
        increment_window = int(increment_volatility_size)
    if increment_window < window:
        raise ValueError(
            "Config error: increment_volatility_size must be greater than or equal to volatility_window_size."
        )
    if compute_confidence_interval:
        if delta_n is None or not np.isfinite(delta_n) or float(delta_n) <= 0:
            raise ValueError(
                "Config error: delta_n must be a positive finite float when compute_confidence_interval=True."
            )
        delta_n = float(delta_n)

    kappa = increment_window / window

    # Parse truncation
    price_trunc_method, price_trunc_param = parse_truncation_mode(price_truncation_mode)
    vol_trunc_method, vol_trunc_param = parse_truncation_mode(volatility_truncation_mode)

    # Step 1
    # print("Step 1/7: Preparing price data...")

    n_day, price_per_day = X.shape
    daily_prices = [X[i, :] for i in range(n_day)]

    # Step 2
    # print("Step 2/7: Computing daily volatility-squared series for each day...")

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
        return None
    
    # print(f"Total log-returns processed: n={n_total}, truncated points: N={N_total}, proportion: p={N_total / n_total if n_total > 0 else 0.0:.6f}")

    # Step 3
    # print("Step 3/7: Normalising and removing intraday volatility pattern if applicable...")

    # Align lengths (pattern removal and averaging require same intraday index)
    min_len = min(v.shape[0] for v in daily_volatility_squared_list)
    max_len = max(v.shape[0] for v in daily_volatility_squared_list)
    # print(f"Volume intensity length range: min={min_len}, max={max_len}")
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
    # print("Step 4/7: Estimating autocorrelation vectors...")

    if N_autocorrelation is None or int(N_autocorrelation) <= 2:
        raise ValueError("Config error: N_autocorrelation must be an integer greater than 2.")
    n_lags = int(N_autocorrelation)

    series_len = daily_volatility_squared_list[0].shape[0]
    if series_len <= n_lags:
        raise ValueError(
            f"Config error: N_autocorrelation ({n_lags}) must be smaller than the volatility series length ({series_len})."
        )

    daily_autocorr_vectors: List[np.ndarray] = []
    n_increments_total = 0
    truncated_increments_total = 0
    LA0 = []
    n_truncated_days = 0
    for vsq in daily_volatility_squared_list:
        autocorr, n_increments, truncated = compute_autocorrelation(
            vsq,
            n_lags,
            increment_window,
            truncation_method=vol_trunc_method,
            truncation_param=vol_trunc_param,
            return_counts=True,
        )
        # autocorr = autocorr / autocorr[0]
        daily_autocorr_vectors.append(autocorr)
        LA0.append(autocorr[0])
        n_increments_total += n_increments
        truncated_increments_total += truncated
        if truncated > 0:
            n_truncated_days += 1

    daily_autocorr_matrix = np.vstack(daily_autocorr_vectors)
    average_autocorrelation = np.mean(daily_autocorr_matrix, axis=0)
    # print(LA0)

    # print(f"Average autocorrelation vector: {average_autocorrelation}")
    if print_truncated_infos:
        proportion = (
            truncated_increments_total / n_increments_total
            if n_increments_total > 0
            else 0.0
        )
        print(
            "Volatility increments truncated: "
            f"{truncated_increments_total} out of {n_increments_total} "
            f"(proportion: {proportion:.6f}) "
            f"(n_truncated_days: {n_truncated_days:.6f})"
        )

    # Step 5: Estimate covariance matrices
    # print("Step 5/7: Estimating covariance matrices...")

    daily_covariance_matrices: List[np.ndarray] = []

    if GMM_weight not in {"identity", "optimal"}:
        raise ValueError(f"Invalid GMM_weight: {GMM_weight}. Expected 'identity' or 'optimal'.")
    if GMM_weight == "optimal" or compute_confidence_interval:
        for vsq in daily_volatility_squared_list:
            daily_covariance_matrices.append(compute_covariance(
                W_fun, 
                Ln,
                Kn, 
                vsq, 
                n_lags, 
                increment_window, 
                truncation_method=vol_trunc_method, 
                truncation_param=vol_trunc_param
            ).flatten())
        daily_covariance_matrices = np.vstack(daily_covariance_matrices)
        average_covariance = np.mean(daily_covariance_matrices, axis=0).reshape((n_lags - 1, n_lags - 1))
        
    # Step 6: GMM estimation of Hurst exponent
    # print(
    #     f"Step 6/7: Estimating Hurst exponent via GMM on grid H in "
    #     f"[{hurst_min_value}, {hurst_max_value}] with step {hurst_step}..."
    # )

    Psi = create_Psi_function(increment_window, n_lags, kappa)

    weight_matrix = np.identity(n_lags - 1)
    if GMM_weight == "optimal":
        weight_matrix = np.linalg.inv(average_covariance)

    H_total, R_total = estimation_GMM(weight_matrix,
                            average_autocorrelation,
                            Psi,
                            hurst_min_value,
                            hurst_max_value,
                            hurst_step,
                            optimization_method=optimization_method)
        
    # Step 7: Save results to output folder and print results as well
    # print("Step 7/7: Saving results and printing summary...")
    # print(f"Estimated Hurst exponent: {H_total}")

    confidence = None
    if compute_confidence_interval:
        n_days = len(daily_volatility_squared_list)

        C1, C2 = get_confidence_size(n_lags, increment_window, kappa, H_total, R_total, n_days, delta_n, average_covariance, weight_matrix)
        alpha = 0.95  # Example: 95% confidence interval
        z_alpha = norm.ppf((1 + alpha) / 2)  # Compute Φ^−1((1 - α) / 2)
        confidence = C1 * z_alpha
        # print("Size confidence band:", C1 * z_alpha)

    return H_total, confidence


def list_prepared_data_years(input_data_folder: str) -> List[Tuple[int, str]]:
    pattern = re.compile(r"^prepared_data_(\d{4})\.npy$")
    years_paths = []
    for name in os.listdir(input_data_folder):
        match = pattern.match(name)
        if match:
            years_paths.append((int(match.group(1)), os.path.join(input_data_folder, name)))
    years_paths.sort(key=lambda x: x[0])
    return years_paths


def run_pipeline(
    input_data_folder: str,
    price_truncation_mode: Optional[str] = None,
    volatility_truncation_mode: Optional[str] = None,
    remove_pattern: Optional[str] = None,
    volatility_window_size: Optional[int] = None,
    increment_volatility_size: Optional[int] = None,
    hurst_min_value: Optional[float] = None,
    hurst_max_value: Optional[float] = None,
    hurst_step: Optional[float] = None,
    delta_n: Optional[float] = None,
    normalise_average_value: bool = True,
    N_autocorrelation: Optional[int] = None,
    compute_confidence_interval: bool = False,
    GMM_weight: str = "identity",
    Ln: int = 180,
    Kn: int = 720,
    W_fun_id: str = "parzen",
    N_consecutive_years: Optional[int] = 0,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    print_truncated_infos: bool = False,
    optimization_method: str = "brute",
) -> Optional[
    Union[
        Tuple[float, Optional[float]],
        Dict[Tuple[int, ...], Optional[Tuple[float, Optional[float]]]],
    ]
]:
    if input_data_folder is None:
        raise ValueError("Config error: input_data_folder is None.")

    if volatility_window_size is None or int(volatility_window_size) <= 0:
        raise ValueError("Config error: volatility_window_size must be a positive integer.")
    window = int(volatility_window_size)

    if increment_volatility_size is None:
        increment_window = window
    else:
        increment_window = int(increment_volatility_size)
    if increment_window < window:
        raise ValueError(
            "Config error: increment_volatility_size must be greater than or equal to volatility_window_size."
        )

    base_dir = os.path.dirname(__file__)
    folder_path = os.path.join(base_dir, input_data_folder)
    years_paths = list_prepared_data_years(folder_path)
    if not years_paths:
        raise ValueError(f"No prepared data files found in {folder_path}.")

    available_start_year = years_paths[0][0]
    available_end_year = years_paths[-1][0]
    effective_start_year = available_start_year if start_year is None else int(start_year)
    effective_end_year = available_end_year if end_year is None else int(end_year)

    if effective_start_year > effective_end_year:
        raise ValueError(
            f"Invalid year range: start_year ({effective_start_year}) > "
            f"end_year ({effective_end_year})."
        )

    years_paths = [
        (year, path)
        for year, path in years_paths
        if effective_start_year <= year <= effective_end_year
    ]
    if not years_paths:
        raise ValueError(
            f"No prepared data files found in {folder_path} for years "
            f"{effective_start_year}-{effective_end_year}."
        )

    data_by_year = {year: np.load(path, allow_pickle=True) for year, path in years_paths}
    years = [year for year, _ in years_paths]

    if N_consecutive_years is None:
        N_consecutive_years = len(years)
    else:
        N_consecutive_years = int(N_consecutive_years)

    if N_consecutive_years <= 0:
        X = np.concatenate([data_by_year[year] for year in years], axis=0)
        return _run_pipeline_from_array(
            X,
            price_truncation_mode=price_truncation_mode,
            volatility_truncation_mode=volatility_truncation_mode,
            remove_pattern=remove_pattern,
            volatility_window_size=volatility_window_size,
            increment_volatility_size=increment_window,
            hurst_min_value=hurst_min_value,
            hurst_max_value=hurst_max_value,
            hurst_step=hurst_step,
            delta_n=delta_n,
            normalise_average_value=normalise_average_value,
            N_autocorrelation=N_autocorrelation,
            compute_confidence_interval=compute_confidence_interval,
            GMM_weight=GMM_weight,
            Ln=Ln,
            Kn=Kn,
            W_fun_id=W_fun_id,
            print_truncated_infos=print_truncated_infos,
            optimization_method=optimization_method,
        )
    


    results: Dict[Tuple[int, ...], Optional[Tuple[float, Optional[float]]]] = {}
    span = N_consecutive_years
    for i in range(0, len(years) - span + 1):
        span_years = years[i:i + span]
        if span_years[-1] - span_years[0] != span - 1:
            continue
        X = np.concatenate([data_by_year[year] for year in span_years], axis=0)
        result = _run_pipeline_from_array(
            X,
            price_truncation_mode=price_truncation_mode,
            volatility_truncation_mode=volatility_truncation_mode,
            remove_pattern=remove_pattern,
            volatility_window_size=volatility_window_size,
            increment_volatility_size=increment_volatility_size,
            hurst_min_value=hurst_min_value,
            hurst_max_value=hurst_max_value,
            hurst_step=hurst_step,
            delta_n=delta_n,
            normalise_average_value=normalise_average_value,
            N_autocorrelation=N_autocorrelation,
            compute_confidence_interval=compute_confidence_interval,
            GMM_weight=GMM_weight,
            Ln=Ln,
            Kn=Kn,
            W_fun_id=W_fun_id,
            print_truncated_infos=print_truncated_infos,
            optimization_method=optimization_method,
        )
        results[tuple(span_years)] = result
        if result is None:
            print(f"Span: {span_years[0]}-{span_years[-1]} H=None")
            continue
        H, confidence = result
        if confidence is None:
            print(f"Span: {span_years[0]}-{span_years[-1]} H={H}")
        else:
            print(f"Span: {span_years[0]}-{span_years[-1]} H={H} +/- {confidence:.4f}")
    return results


if __name__ == "__main__":
    run_pipeline(
        input_data_folder=input_data_folder,
        price_truncation_mode=price_truncation_mode,
        volatility_truncation_mode=volatility_truncation_mode,
        remove_pattern=remove_pattern,
        volatility_window_size=volatility_window_size,
        increment_volatility_size=increment_volatility_size,
        hurst_min_value=hurst_min_value,
        hurst_max_value=hurst_max_value,
        hurst_step=hurst_step,
        normalise_average_value=normalise_average_value,
        N_autocorrelation=N_autocorrelation,
        compute_confidence_interval=compute_confidence_interval,
        GMM_weight=GMM_weight,
        Ln=Ln,
        Kn=Kn,
        W_fun_id=W_fun_id,
        delta_n=5.0/(252.0 * 23400.0),
        N_consecutive_years=N_consecutive_years,
        start_year=start_year,
        end_year=end_year,
        print_truncated_infos=print_truncated_infos,
        optimization_method=optimization_method,
    )
