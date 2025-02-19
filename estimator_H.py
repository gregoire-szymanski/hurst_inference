import numpy as np

##### Preliminary functions

def Phi_Hl(l, H):
    """
    Compute the value of Phi^H_ell based on the given formula.

    :param l: The parameter ell in the formula
    :param H: The parameter H (Hurst exponent) in the formula
    :return: The computed value of Phi^H_ell
    """
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    return numerator / denominator





def dPhi_Hl_dH(l, H):
    """
    Compute the derivative of Phi^H_ell with respect to H.

    :param l: The parameter ell in the formula
    :param H: The parameter H (Hurst exponent) in the formula
    :return: The computed derivative d/dH (Phi^H_ell)
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


def dichotomic_search(f, target, low, high, is_increasing=True, epsilon=1e-5):
    """
    Perform a dichotomic (binary) search for a target value within a specified interval [low, high]
    of a monotonic function 'f'.

    :param f: Monotonic function to search over.
    :param target: Target value to search for.
    :param low: Lower bound of the search interval.
    :param high: Upper bound of the search interval.
    :param is_increasing: True if the function is increasing, False if decreasing.
    :param epsilon: Tolerance for the convergence of the search.
    :return: The point where the function value is closest to the target, or None if not found.
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


##### Ratio with different windows

def ratio_estimator(QV, QV_2kn):
    return np.log(QV_2kn / QV) / (np.log(2) * 2)

##### Ratio with different lags

def ratio_2_01(H):
    return Phi_Hl(2, H) / (Phi_Hl(0, H) + 2 * Phi_Hl(1, H))

def inverse_ratio_2_01(target):
    return dichotomic_search(ratio_2_01, target, 1e-5, 1, is_increasing=True, epsilon=1e-5)

def estimation_01_2(QV01, QV2):
    return inverse_ratio_2_01(QV2 / QV01)


##### GMM estimator

# GMM method tries to minimize 
# F(H,R) = (V-P)^T W (V-P)
# F(H,R) = V^T W V - P^T W V - V^T W P + P^T W P

def F_estimation_GMM(W, V, Psi_func, H, normalisation = 1):
    # Extract the scalar H from the provided list or array
    H = H[0]
    
    # Ensure V is a column matrix
    # If V is already 2D with shape (n,1), this will do nothing.
    # If V is 1D (shape (n,)), it will reshape into (n,1).
    V = np.atleast_2d(V).reshape(-1, 1)
    
    # Compute Psi as a column vector
    Psi = Psi_func(H)
    Psi = np.atleast_2d(Psi).reshape(-1, 1)
    
    # Compute terms:
    # term0 = Vᵀ W V
    # term1 = (Psiᵀ W V) + (Vᵀ W Psi)
    # term2 = Psiᵀ W Psi
    
    term0 = V.T @ W @ V
    term1 = (Psi.T @ W @ V) + (V.T @ W @ Psi)
    term2 = Psi.T @ W @ Psi
    
    # Since these are all (1x1) results, extract the scalar values
    term0 = term0[0, 0]
    term1 = term1[0, 0]
    term2 = term2[0, 0]
    
    R = term1 / term2 / 2
    
    # Return the scalar result:
    return normalisation * (term0 - R * term1 + term2 * R * R)


def estimation_GMM(W, V, Psi_func, H_min=0.001, H_max=0.499, mesh=0.001, debug=False):
    # Create a grid of H values
    H_values = np.arange(H_min, H_max, mesh)
    
    # Evaluate F_estimation_GMM for each H in the grid
    F_values = [F_estimation_GMM(W, V, Psi_func, [H]) for H in H_values]
    
    # Find the index of the minimum value
    min_index = np.argmin(F_values)
    
    if debug:
        return H_values, F_values, min_index
    # Return the H that gives the smallest F-estimation
    return H_values[min_index]





