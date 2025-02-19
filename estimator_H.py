import numpy as np

##### Preliminary functions

def Phi_Hl(l: int, H: float) -> float:
    """
    Compute the value of \(\Phi^H_\ell\) using a finite difference formula.

    This function evaluates a discrete approximation based on powers of absolute values,
    commonly used in fractional Brownian motion and related models.

    :param l: Index \(\ell\) in the formula (integer).
    :param H: Hurst exponent \(H\), controlling the memory effect (float).
    :return: Computed value of \(\Phi^H_\ell\).
    """
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    return numerator / denominator




def dPhi_Hl_dH(l: int, H: float) -> float:
    """
    Compute the derivative of \(\Phi^H_\ell\) with respect to \(H\).

    Uses the chain rule to differentiate power terms in the finite difference formula.

    :param l: Index \(\ell\) in the formula (integer).
    :param H: Hurst exponent \(H\) (float).
    :return: The computed derivative \(\frac{d}{dH} \Phi^H_\ell\).
    """
    def power_term_derivative(x, H):
        return (2 * x ** (2 * H + 2) * np.log(np.abs(x)))
    
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))

    numerator_derivative = (
        power_term_derivative(np.abs(l + 2), H) - 4 * power_term_derivative(np.abs(l + 1), H) +
        6 * power_term_derivative(np.abs(l), H) - 4 * power_term_derivative(np.abs(l - 1), H) +
        power_term_derivative(np.abs(l - 2), H)
    )
    
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    denominator_derivative = 4 * (4 * H + 3)
    
    return (numerator_derivative * denominator - denominator_derivative * numerator) / (denominator * denominator)


def dichotomic_search(f, target: float, low: float, high: float, is_increasing: bool = True, epsilon: float = 1e-5) -> float:
    """
    Perform a dichotomic (binary) search to find an approximate solution \(x\) such that \(f(x) \approx \text{target}\).

    The function `f` is assumed to be monotonic (either increasing or decreasing, default being increasing).
    
    :param f: Monotonic function to search over.
    :param target: Target value to search for.
    :param low: Lower bound of the search interval.
    :param high: Upper bound of the search interval.
    :param is_increasing: If True, `f` is increasing; otherwise, it is decreasing.
    :param epsilon: Tolerance for stopping the search.
    :return: Approximate solution \(x\) where \(f(x) \approx \text{target}\).
    """

    if is_increasing:
        if f(low) > target:
            return low
        if f(high) < target:
            return high
    else:
        if f(low) < target:
            return low
        if f(high) > target:
            return high

    while low <= high:
        mid = (low + high) / 2
        f_mid = f(mid)

        if abs(f_mid - target) < epsilon:
            return mid  # Found a value close to the target

        if is_increasing:
            if f_mid < target:
                low = mid + epsilon
            else:
                high = mid - epsilon
        else:
            if f_mid > target:
                low = mid + epsilon
            else:
                high = mid - epsilon

    return None  # Target value not found within the interval


##### GMM estimator

def F_estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H: list, normalisation: float = 1) -> float:
    """
    Compute the GMM objective function \(F(H, R)\) for given parameters.
    
    This function minimizes:
    
    \[ F(H, R) = (V - P)^T W (V - P) \]
    
    where \(P\) is computed based on \(H\).

    :param W: Weight matrix (numpy array).
    :param V: Observation vector (numpy array).
    :param Psi_func: Function \(\Psi(H)\) providing model predictions.
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


def estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H_min: float = 0.001, H_max: float = 0.499, mesh: float = 0.001, debug: bool = False):
    """
    Perform Generalized Method of Moments (GMM) estimation for the Hurst exponent.
    
    This method finds \(H\) that minimizes the GMM objective function over a predefined grid.
    
    :param W: Weight matrix (numpy array).
    :param V: Observation vector (numpy array).
    :param Psi_func: Function returning model predictions \(\Psi(H)\).
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
        return H_values, F_values, min_index

    return H_values[min_index]





