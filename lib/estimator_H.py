import numpy as np

from scipy.special import gamma

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




def dPhi_Hl_dH(l: int, H: float) -> float:
    """
    Compute the derivative of $\\Phi^H_\\ell$ with respect to $H$.

    Uses the chain rule to differentiate power terms in the finite difference formula.

    :param l: Index $\\ell$ in the formula (integer).
    :param H: Hurst exponent $H$ (float).
    :return: The computed derivative $\\frac{d}{dH} \\Phi^H_\\ell$.
    """
    def power_term_derivative(x, H):
        if x == 0:
            return 0
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
    Perform a dichotomic (binary) search to find an approximate solution $x$ such that $f(x) \\approx \\text{target}$.

    The function `f` is assumed to be monotonic (either increasing or decreasing, default being increasing).
    
    :param f: Monotonic function to search over.
    :param target: Target value to search for.
    :param low: Lower bound of the search interval.
    :param high: Upper bound of the search interval.
    :param is_increasing: If True, `f` is increasing; otherwise, it is decreasing.
    :param epsilon: Tolerance for stopping the search.
    :return: Approximate solution $x$ where $f(x) \\approx \\text{target}$.
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



def uncorrected_alpha(theta, lag, H):
    return theta**(2*H-1) * dPhi_Hl_dH(lag, H) + 2 * np.log(theta) * Phi_Hl(lag, H)

def uncorrected_beta(theta, lag, H):
    return theta**(2*H-1) * Phi_Hl(lag, H)

def variation_44(f):
    shifts = [-2, -1, 0, 1, 2]
    coefficients = [1, -4, 6, -4, 1]
    return sum([
        c1 * c2 * f(s1, s2)
        for (s1, c1) in zip(shifts, coefficients)
        for (s2, c2) in zip(shifts, coefficients)
    ])
        

def uncorrected_gamma(theta1, theta2, lag1, lag2, H):
    if H == 0.25:
        local_f = lambda l1, l2: np.abs((l1 + lag1) * theta1 + (l2 + lag2) * theta2)**(6) * np.log(np.abs((l1 + lag1) * theta1 + (l2 + lag2) * theta2)) + np.abs((l1 + lag1) * theta1 - (l2 + lag2) * theta2)**(6) * np.log(np.abs((l1 + lag1) * theta1 - (l2 + lag2) * theta2))
        return variation_44(local_f) / ( 5760 * (theta1 * theta2)**3)
    else:
        local_f = lambda l1, l2: np.abs((l1 + lag1) * theta1 + (l2 + lag2) * theta2)**(4 * H + 5) + np.abs((l1 + lag1) * theta1 - (l2 + lag2) * theta2)**(4 * H + 5)
        return gamma(1+2*H)**2 * (1-1/np.cos(2*np.pi * H)) * variation_44(local_f) / ( 4 * gamma(6+4*H) * (theta1 * theta2)**(2*H + 5))



def compute_alpha(theta, lag, H):
    if lag == 1:
        return uncorrected_alpha(theta, 0, H) + 2 * uncorrected_alpha(theta, 1, H)
    return uncorrected_alpha(theta, lag, H)


def compute_beta(theta, lag, H):
    if lag == 1:
        return uncorrected_beta(theta, 0, H) + 2 * uncorrected_beta(theta, 1, H)
    return uncorrected_beta(theta, lag, H)


def compute_gamma(theta1, theta2, lag1, lag2, H):
    if lag1 == 1 and lag2 == 1:
        return uncorrected_gamma(theta1, theta2, 0, 0, H) + 2 * uncorrected_gamma(theta1, theta2, 0, 1, H) + 2 * uncorrected_gamma(theta1, theta2, 1, 0, H) + 4 * uncorrected_gamma(theta1, theta2, 1, 1, H)
    elif lag1 == 1 and lag2 > 1:
        return uncorrected_gamma(theta1, theta2, 0, lag2, H) + 2 * uncorrected_gamma(theta1, theta2, 1, lag2, H) 
    elif lag1 > 1 and lag2 == 1:
        return uncorrected_gamma(theta1, theta2, lag1, 0, H) + 2 * uncorrected_gamma(theta1, theta2, lag1, 1, H) 
    else:
        return uncorrected_gamma(theta1, theta2, lag1, lag2, H)
    
def get_optimal_variance(params_volatility, H, eta, n_days, delta_n):
    R = eta * eta * n_days / 252

    window_values = []
    lags_values = []

    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']
        
        window_values.extend([window for _ in range(N_lags)])
        lags_values.extend([i for i in range(1,N_lags+1)])

    reference_window = window_values[0]
    theta = [w / reference_window for w in window_values]

    m = len(window_values)


    alpha = np.zeros(m)
    beta = np.zeros(m)

    for i in range(m):
        alpha[i] = compute_alpha(theta[i], lags_values[i], H)
        beta[i] = compute_beta(theta[i], lags_values[i], H)

    alpha_beta = np.array([alpha, beta])

    Sigma = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            Sigma[i,j] = compute_gamma(theta[i], theta[j], lags_values[i], lags_values[j], H) * (theta[i] * theta[j])**(2*H-1/2)

    u_t = np.array([alpha * R, beta]).transpose()

    D = np.array([
        [1, 0],
        [-2 * np.log(reference_window * delta_n), 1]
    ])

    W = np.linalg.inv(Sigma)

    uWu_inv = np.linalg.inv(u_t.transpose() @ W @ u_t)
    matrix_43 = reference_window * delta_n * D @ uWu_inv @ u_t.transpose() @ W @ Sigma @ W @ u_t @ uWu_inv @ D.transpose()

    return matrix_43[0,0]**0.5, matrix_43[1,1]**0.5





def get_confidence_size(params_volatility, H_estimated, R_estimated, n_days, delta_n, Sigma_estimated, W_chosen):
    window_values = []
    lags_values = []

    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']
        
        window_values.extend([window for _ in range(N_lags)])
        lags_values.extend([i for i in range(1,N_lags+1)])

    reference_window = window_values[0]
    theta = [w / reference_window for w in window_values]

    m = len(window_values)


    alpha = np.zeros(m)
    beta = np.zeros(m)

    for i in range(m):
        alpha[i] = compute_alpha(theta[i], lags_values[i], H_estimated)
        beta[i] = compute_beta(theta[i], lags_values[i], H_estimated)

    alpha_beta = np.array([alpha, beta])

    u_t = np.array([alpha * R_estimated, beta]).transpose()

    D = np.array([
        [1, 0],
        [-2 * np.log(reference_window * delta_n), 1]
    ])

    uWu_inv = np.linalg.inv(u_t.transpose() @ W_chosen @ u_t)
    matrix_43 = (delta_n * reference_window)**(1-4*H_estimated) * reference_window * delta_n * D @ uWu_inv @ u_t.transpose() @ W_chosen @ Sigma_estimated @ W_chosen @ u_t @ uWu_inv @ D.transpose()
    # matrix_43 = reference_window * delta_n * D @ uWu_inv @ u_t.transpose() @ W_chosen @ Sigma_estimated @ W_chosen @ u_t @ uWu_inv @ D.transpose()

    return matrix_43[0,0]**0.5, matrix_43[1,1]**0.5

