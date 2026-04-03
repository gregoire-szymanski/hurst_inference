from __future__ import annotations

from math import comb, cos, gamma, isclose, log, pi
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _H_alpha(x: float, alpha: float) -> float:
    """Return |x|^alpha with the convention H_alpha(0)=0."""
    ax = abs(x)
    if ax == 0.0:
        return 0.0
    return ax ** alpha


def _H6_log(x: float) -> float:
    """Return |x|^6 log|x| with the convention H(0)=0."""
    ax = abs(x)
    if ax == 0.0:
        return 0.0
    return (ax ** 6.0) * log(ax)


def delta_h(phi: Callable[[float], float], x: float, h: float, n: int) -> float:
    """
    Central finite difference:
        delta_h^n phi(x) = sum_{k=0}^n (-1)^k C(n,k) phi(x + (n/2 - k)h).
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    half = 0.5 * n
    total = 0.0
    for k in range(n + 1):
        total += ((-1.0) ** k) * comb(n, k) * phi(x + (half - k) * h)
    return total


def delta_h1_h2(phi: Callable[[float], float], x: float, h1: float, h2: float, n: int) -> float:
    """
    Mixed central finite difference for even n:
        delta_{h1,h2}^n phi(x) = delta_{h1}^{n/2}[delta_{h2}^{n/2} phi](x).
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be an even integer >= 2")

    half_n = n // 2

    def inner(y: float) -> float:
        return delta_h(phi, y, h2, half_n)

    return delta_h(inner, x, h1, half_n)


def _delta8_1_1_over_kappa(phi: Callable[[float], float], x: float, kappa: float) -> float:
    return delta_h1_h2(phi, x, 1.0, 1.0 / kappa, 8)


def R_kappa_e(l: int, l_prime: int, e: float, kappa: float = 1.0) -> float:
    """
    Compute R_{kappa,e}(l,l'):

    R_{kappa,e}(l,l') =
        [kappa^4 * Gamma(2e+2)^2 * (1 + 1/cos(2*pi*e))] / [4*Gamma(4e+8)]
        * (delta^8_{1,1/kappa} H_{4e+7}(l+l') + delta^8_{1,1/kappa} H_{4e+7}(l-l')).

    Uses the continuous extension at e = -1/4:
        R_{kappa,-1/4}(l,l') = (kappa^4 / 5760)
        * (delta^8_{1,1/kappa} H6_log(l+l') + delta^8_{1,1/kappa} H6_log(l-l')).
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive")

    x_plus = float(l + l_prime)
    x_minus = float(l - l_prime)

    if isclose(e, -0.25, rel_tol=0.0, abs_tol=1e-14):
        prefactor = (kappa ** 4.0) / 5760.0
        return prefactor * (
            _delta8_1_1_over_kappa(_H6_log, x_plus, kappa)
            + _delta8_1_1_over_kappa(_H6_log, x_minus, kappa)
        )

    cos_term = cos(2.0 * pi * e)
    if isclose(cos_term, 0.0, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("cos(2*pi*e) is numerically zero; use e=-1/4 limit when appropriate.")

    alpha = 4.0 * e + 7.0
    prefactor = (
        (kappa ** 4.0)
        * (gamma(2.0 * e + 2.0) ** 2.0)
        * (1.0 + 1.0 / cos_term)
        / (4.0 * gamma(4.0 * e + 8.0))
    )

    def h_alpha(y: float) -> float:
        return _H_alpha(y, alpha)

    return prefactor * (
        _delta8_1_1_over_kappa(h_alpha, x_plus, kappa)
        + _delta8_1_1_over_kappa(h_alpha, x_minus, kappa)
    )


def calR_kappa_e(l: int, l_prime: int, e: float, kappa: float = 1.0) -> float:
    """
    Piecewise definition of \\mathcal{R}_{kappa,e}(l,l') from Eq. (R).
    Indices l and l_prime follow the paper convention and must be >= 1.
    """
    if l < 1 or l_prime < 1:
        raise ValueError("l and l_prime must be >= 1")

    if l >= 2 and l_prime >= 2:
        return R_kappa_e(l, l_prime, e, kappa)
    if l == 1 and l_prime >= 2:
        return R_kappa_e(0, l_prime, e, kappa) + 2.0 * R_kappa_e(1, l_prime, e, kappa)
    if l >= 2 and l_prime == 1:
        return R_kappa_e(l, 0, e, kappa) + 2.0 * R_kappa_e(l, 1, e, kappa)

    return (
        R_kappa_e(0, 0, e, kappa)
        + 4.0 * R_kappa_e(0, 1, e, kappa)
        + 4.0 * R_kappa_e(1, 1, e, kappa)
    )


def compute_covariance_matrix_from_hurst(kappa: float, H: float, n_lags: int) -> np.ndarray:
    """
    Build the covariance matrix using calR_{kappa,e}(l,l') for l,l' in {1,...,n_lags}.

    In the notation of your formulas, e = H - 1/2.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive")
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")

    e = H - 0.5
    covariance = np.zeros((n_lags, n_lags), dtype=float)

    for i in range(1, n_lags + 1):
        for j in range(i, n_lags + 1):
            value = calR_kappa_e(i, j, e, kappa)
            covariance[i - 1, j - 1] = value
            covariance[j - 1, i - 1] = value

    return covariance


def covariance_to_correlation(covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert covariance matrix to correlation matrix and return:
      - correlation matrix
      - normalized covariance diagonal diag(cov) / diag(cov)[0]
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")

    diagonal_covariance = np.diag(covariance).astype(float)
    inv_std = np.zeros_like(diagonal_covariance)
    np.divide(1.0, np.sqrt(diagonal_covariance), out=inv_std, where=diagonal_covariance > 0.0)
    correlation = covariance * inv_std[:, None] * inv_std[None, :]

    diagonal_normalized = diagonal_covariance.copy()
    if diagonal_normalized.size > 0 and np.isfinite(diagonal_normalized[0]) and diagonal_normalized[0] != 0.0:
        diagonal_normalized = diagonal_normalized / diagonal_normalized[0]
    else:
        diagonal_normalized[:] = np.nan

    return correlation, diagonal_normalized


def compute_correlation_matrix_from_hurst(kappa: float, H: float, n_lags: int) -> np.ndarray:
    """Convenience wrapper returning only the correlation matrix."""
    covariance = compute_covariance_matrix_from_hurst(kappa=kappa, H=H, n_lags=n_lags)
    correlation, _ = covariance_to_correlation(covariance)
    return correlation


def plot_correlation_matrix(
    kappa: float,
    H: float,
    n_lags: int,
    title: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Compute covariance/correlation matrices from (kappa, H) and plot them.

    Returns (covariance, correlation, figure).
    """
    average_covariance = compute_covariance_matrix_from_hurst(kappa=kappa, H=H, n_lags=n_lags)
    average_correlation, diagonal_covariance = covariance_to_correlation(average_covariance)

    n_rows, n_cols = average_correlation.shape
    fig = plt.figure(figsize=(8.3, 6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, n_cols], wspace=0.08)
    ax_diag = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1], sharey=ax_diag)

    finite_diagonal = diagonal_covariance[np.isfinite(diagonal_covariance)]
    if finite_diagonal.size > 0:
        diag_min = float(np.min(finite_diagonal))
        diag_max = float(np.max(finite_diagonal))
        if diag_min == diag_max:
            padding = abs(diag_min) * 0.1 if diag_min != 0.0 else 1.0
            diag_min -= padding
            diag_max += padding
    else:
        diag_min, diag_max = -1.0, 1.0

    diagonal_column = diagonal_covariance[:, None]
    ax_diag.imshow(diagonal_column, cmap="YlGnBu", aspect="auto", vmin=diag_min, vmax=diag_max)
    diagonal_span = diag_max - diag_min
    for row in range(n_rows):
        value = diagonal_covariance[row]
        if np.isfinite(value) and diagonal_span > 0.0:
            color_position = (value - diag_min) / diagonal_span
        else:
            color_position = 0.0
        text_color = "white" if color_position > 0.55 else "black"
        ax_diag.text(
            0,
            row,
            f"{value:.3g}" if np.isfinite(value) else "nan",
            ha="center",
            va="center",
            color=text_color,
            fontsize=8,
            fontweight="bold",
        )

    image = ax.imshow(average_correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    for row in range(n_rows):
        for col in range(n_cols):
            value = average_correlation[row, col]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    lag_labels = np.arange(1, n_rows + 1)

    ax_diag.set_title("diag(Cov)")
    ax_diag.set_xticks([])
    ax_diag.set_yticks(np.arange(n_rows))
    ax_diag.set_yticklabels(lag_labels)
    ax_diag.set_ylabel("Lag index")
    for spine_name in ("top", "right", "bottom"):
        ax_diag.spines[spine_name].set_visible(False)

    matrix_title = "Normalised average covariance matrix"
    if title is not None:
        matrix_title = title

    ax.set_title(matrix_title)
    ax.set_xlabel("Lag index")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(lag_labels)
    ax.tick_params(axis="y", left=False, labelleft=False)
    fig.colorbar(image, ax=ax, label="Correlation")

    if show:
        plt.show()

    return average_covariance, average_correlation, fig


plot_correlation_matrix(
    kappa=60/48,
    H=0.3,
    n_lags=6,
    # title: Optional[str] = None,
    show=True,
)