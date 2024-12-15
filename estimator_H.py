import numpy as np

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


def ratio_2_01(H):
    return Phi_Hl(2, H) / (Phi_Hl(0, H) + 2 * Phi_Hl(1, H))

def inverse_ratio_2_01(target):
    return dichotomic_search(ratio_2_01, target, 1e-5, 1, is_increasing=True, epsilon=1e-5)

def ratio_estimator(QV, QV_2kn):
    return np.log(QV_2kn / QV) / (np.log(2) * 2)