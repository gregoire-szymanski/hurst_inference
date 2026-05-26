import argparse
import csv
import json
import os
import re
import time
from typing import List, Optional, Tuple
from scipy.stats import norm

import numpy as np

input_data_folder = "clean_data/"
# input_data_folder = "short_data/"

N_years_backtest = 1

price_truncation_mode = 'BIVAR_3'  # None, STD_X, BIVAR_X (X int/float)
volatility_truncation_mode = 'STD_3'  # None, STD_X (X int/float)
remove_pattern = 'multiplicative'  # None, multiplicative, additive

volatility_window_size = 60 * 3 # Integer

normalise_average_value = True  # True or False, default True

N_autocorrelation = 6  # Integer

_args = {}
_global_args = globals().get("args")
if isinstance(_global_args, dict):
    _args.update(_global_args)

_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument("--N", type=int)
_cli_parser.add_argument("--volatility_window_size", type=int)
_cli_args, _ = _cli_parser.parse_known_args()
if _cli_args.N is not None:
    _args["N"] = _cli_args.N
if _cli_args.volatility_window_size is not None:
    _args["volatility_window_size"] = _cli_args.volatility_window_size

if "N" in _args and _args["N"] is not None:
    N_autocorrelation = int(_args["N"])
if "volatility_window_size" in _args and _args["volatility_window_size"] is not None:
    volatility_window_size = int(_args["volatility_window_size"])

evaluation_output_csv = f"output/prediction_backtest_results_{volatility_window_size}_{N_autocorrelation}.csv"

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
            autocorr[lag] = np.mean((vol_squared_increments - mean_vol) ** 2)
        else:
            autocorr[lag] = np.mean((vol_squared_increments[lag*window:] - mean_vol) * (vol_squared_increments[:-lag*window] - mean_vol))

    autocorr[1] = autocorr[0] + 2 * autocorr[1]

    if return_counts:
        return autocorr[1:], n_increments, truncated
    return autocorr[1:]

def compute_truncated_volatility_increments(
    vol_squared,
    window,
    truncation_method=None,
    truncation_param=None,
):
    vol_squared = np.asarray(vol_squared, dtype=float)

    vol_squared_increments = vol_squared[window:] - vol_squared[:-window]

    if truncation_method is not None:
        if truncation_method == 'STD':
            std_dev = np.std(vol_squared_increments)
            threshold = float(truncation_param) * std_dev
            vol_squared_increments = truncate_absolute(vol_squared_increments, threshold)
        else:
            raise ValueError(f"Unknown truncation method: {truncation_method}")

    return vol_squared_increments

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

def load_optional_day_index(
    input_folder: str,
    base_name: str,
    expected_days: int,
) -> Optional[List[str]]:
    """Load optional day-index metadata for a prepared_data_YYYY file."""
    candidates = [
        f"{base_name}_dates.npy",
        f"{base_name}_dates.json",
        f"{base_name}_dates.csv",
        f"{base_name}_dates.txt",
    ]
    for candidate in candidates:
        path = os.path.join(input_folder, candidate)
        if not os.path.isfile(path):
            continue
        try:
            if path.endswith(".npy"):
                values = np.asarray(np.load(path, allow_pickle=True)).reshape(-1)
                labels = [str(v) for v in values]
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if not isinstance(loaded, list):
                    continue
                labels = [str(v) for v in loaded]
            else:
                with open(path, "r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    labels = [row[0].strip() for row in reader if row and row[0].strip()]

            if len(labels) == expected_days:
                return labels
        except Exception:
            continue
    return None

def build_day_index_labels(
    data_files: List[str],
    days_per_file: List[int],
    input_folder: str,
) -> List[str]:
    """Build day indices; use date metadata if available, fallback to file-index labels."""
    labels: List[str] = []
    for name, expected_days in zip(data_files, days_per_file):
        base_name, _ = os.path.splitext(name)
        file_labels = load_optional_day_index(input_folder, base_name, expected_days)
        if file_labels is not None:
            labels.extend(file_labels)
            continue

        match = re.search(r"(\d{4})", name)
        year = match.group(1) if match else base_name
        labels.extend([f"{year}_day_{day_idx + 1:03d}" for day_idx in range(expected_days)])
    return labels

def save_backtest_results_csv(
    output_csv_path: str,
    rows: List[List[object]],
    n_params: int,
) -> None:
    header = ["start_training", "end_training", "testing", "mse", "mae", "r2", "aic", "aicc", "bic"]
    header.extend([f"p_{i + 1}" for i in range(n_params)])

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def compute_information_criteria(
    residuals: np.ndarray,
    n_params: int,
    n_train_days: int,
) -> Tuple[float, float, float]:
    """Return training-window AIC, AICc, and BIC for the linear predictor.

    AICc uses n - k * n_train_days for the finite-sample correction.
    """
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    n = int(residuals.shape[0])
    k = int(n_params)
    rss = float(np.sum(residuals ** 2))

    if n <= 0 or k < 0 or not np.isfinite(rss):
        return np.nan, np.nan, np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        log_mse = float(np.log(rss / n))

    aic = float(n * log_mse + 2 * k)
    bic = float(n * log_mse + k * np.log(n))
    aicc = float(aic + (2 * k * (k + 1)) / (n - k * int(n_train_days) - 1))

    return aic, aicc, bic




def run_pipeline(
    input_data_folder: str,
    price_truncation_mode: Optional[str] = None,
    volatility_truncation_mode: Optional[str] = None,
    remove_pattern: Optional[str] = None,
    volatility_window_size: Optional[int] = None,
    normalise_average_value: bool = True,
    N_autocorrelation: Optional[int] = None,
    N_years_backtest: int = 4,
    output_results_csv: str = evaluation_output_csv,
) -> Optional[float]:
    """Run the end-to-end Hurst inference pipeline on saved price arrays.

    Loads prices from a folder of NumPy arrays, computes volatility-squared
    series with optional truncation, trains a linear predictor on rolling
    windows, and reports aggregated metrics.
    """
    
    if input_data_folder is None:
        raise ValueError("Config error: input_data_folder is None.")
    if output_results_csv is None or str(output_results_csv).strip() == "":
        raise ValueError("Config error: output_results_csv must be a non-empty path.")

    if os.path.isabs(output_results_csv):
        output_csv_path = output_results_csv
    else:
        output_csv_path = os.path.join(os.path.dirname(__file__), output_results_csv)

    if volatility_window_size is None or int(volatility_window_size) <= 0:
        raise ValueError("Config error: volatility_window_size must be a positive integer.")
    window = int(volatility_window_size)

    # Parse truncation
    price_trunc_method, price_trunc_param = parse_truncation_mode(price_truncation_mode)
    vol_trunc_method, vol_trunc_param = parse_truncation_mode(volatility_truncation_mode)

    # Step 1
    # print("Step 1/7: Listing files, filtering by prefix+date format, loading prices, applying filters...")

    input_data_folder = os.path.join(os.path.dirname(__file__), input_data_folder)
    filenames = [
        name for name in sorted(os.listdir(input_data_folder))
        if os.path.isfile(os.path.join(input_data_folder, name))
    ]
    pattern = re.compile(r"^prepared_data_\d{4}\.npy$")
    date_index_pattern = re.compile(r"^prepared_data_\d{4}_dates\.(npy|json|csv|txt)$")
    invalid_files = [
        name
        for name in filenames
        if not pattern.match(name) and not date_index_pattern.match(name)
    ]
    if invalid_files:
        raise ValueError(f"Unexpected files in input_data_folder: {invalid_files}")

    data_files = [name for name in filenames if pattern.match(name)]
    if not data_files:
        raise ValueError("No prepared_data_YYYY.npy files found in input_data_folder.")

    arrays = []
    days_per_file = []
    for name in data_files:
        data = np.load(os.path.join(input_data_folder, name), allow_pickle=True)
        arrays.append(data)
        days_per_file.append(int(data.shape[0]))

    day_index_labels = build_day_index_labels(data_files, days_per_file, input_data_folder)

    avg_number_days_per_file = float(np.mean(days_per_file))
    N_consecutive_days = int(round(N_years_backtest * avg_number_days_per_file))

    X = np.concatenate(arrays, axis=0)
    n_day, price_per_day = X.shape
    daily_prices = X

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
    # print("Step 3/7: Normalising average values if applicable...")

    min_len = min(v.shape[0] for v in daily_volatility_squared_list)
    max_len = max(v.shape[0] for v in daily_volatility_squared_list)
    # print(f"Volume intensity length range: min={min_len}, max={max_len}")
    daily_vsq = np.stack([v[:min_len] for v in daily_volatility_squared_list])

    if normalise_average_value:
        means = np.mean(daily_vsq, axis=1)
        valid_means = (means != 0.0) & np.isfinite(means)
        daily_vsq[valid_means] = daily_vsq[valid_means] / means[valid_means, None]

    rp = None
    if remove_pattern is not None:
        rp = str(remove_pattern).strip().lower()
        if rp not in ("multiplicative", "additive"):
            raise ValueError(f"Invalid remove_pattern: {remove_pattern}. Expected None, 'multiplicative', or 'additive'.")
        if rp == "additive":
            raise ValueError("remove_pattern='additive' is not supported.")

    # Step 4: Rolling train/test backtest
    # print("Step 4/7: Running rolling train/test backtest...")

    if N_autocorrelation is None or int(N_autocorrelation) < 1:
        raise ValueError("Config error: N_autocorrelation must be an integer greater than 1.")
    n_lags = int(N_autocorrelation)
    offset = window * n_lags
    
    MAX_OFFSET = window * 10  # To ensure we have the same data points for predictors for all choices of n_lags up to 10.

    if MAX_OFFSET > offset:
        print("Warning: The current configuration may lead to fewer training samples due to the offset.")

    n_days = daily_vsq.shape[0]
    print(f"Total number of days: {n_days}")
    last_train_start = n_days - (N_consecutive_days + 2)
    if last_train_start < 0:
        raise ValueError("Not enough days to run the backtest.")

    def build_predictor_matrix(vol_days: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        predictor_rows: List[np.ndarray] = []
        target_rows: List[np.ndarray] = []
        
        for vol_day in np.atleast_2d(vol_days):
            vol_squared_increments = compute_truncated_volatility_increments(
                vol_day,
                window,
                truncation_method=vol_trunc_method,
                truncation_param=vol_trunc_param,
            )
            series_len = vol_squared_increments.shape[0]
            max_k = series_len - offset
            if max_k <= 0:
                continue
            base = np.arange(max(MAX_OFFSET - offset, 0), max_k)
        
            cols = [vol_squared_increments[base + i * window] for i in range(n_lags)]
            predictor_rows.append(np.stack(cols, axis=1))
            target_rows.append(vol_squared_increments[base + offset])

        if not predictor_rows:
            return None
        

        total_predictor_rows = sum(rows.shape[0] for rows in predictor_rows)
        predictor_shape = predictor_rows[0].shape
        X = np.empty((total_predictor_rows, predictor_shape[1]), dtype=predictor_rows[0].dtype)
        start = 0
        for rows in predictor_rows:
            end = start + rows.shape[0]
            X[start:end] = rows
            start = end

        total_target_rows = sum(rows.shape[0] for rows in target_rows)
        y = np.empty((total_target_rows,), dtype=target_rows[0].dtype)
        start = 0
        for rows in target_rows:
            end = start + rows.shape[0]
            y[start:end] = rows
            start = end

        return X, y

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_count = 0
    y_sum = 0.0
    y_sum_sq = 0.0
    n_windows = 0
    rows_for_csv: List[List[object]] = []

    for i in range(last_train_start + 1):
        start_time = time.perf_counter()
        print(f"Training window {i + 1}/{last_train_start + 1}...")
        train_start = i
        train_end = i + N_consecutive_days
        test_idx = i + N_consecutive_days + 1

        train_days = daily_vsq[train_start:train_end + 1]
        test_day = daily_vsq[test_idx]

        if rp == "multiplicative":
            pattern = np.mean(train_days, axis=0)
            pattern = np.where(pattern == 0.0, 1.0, pattern)
            train_days = train_days / pattern
            test_day = test_day / pattern


        train_data = build_predictor_matrix(train_days)
        test_data = build_predictor_matrix(test_day)
        if train_data is None or test_data is None:
            continue




        X_train, y_train = train_data
        X_test, y_test = test_data

        coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)

        train_residual = y_train - (X_train @ coeffs)
        aic, aicc, bic = compute_information_criteria(
            train_residual,
            n_params=n_lags,
            n_train_days=int(train_days.shape[0]),
        )

        y_pred = X_test @ coeffs
        residual = y_test - y_pred
        window_mse = float(np.mean(residual ** 2))
        window_mae = float(np.mean(np.abs(residual)))
        window_var_y = float(np.var(y_test))
        window_r2 = 1.0 - (window_mse / window_var_y) if window_var_y > 0.0 else np.nan

        coeffs_row = np.zeros(n_lags, dtype=float)
        take = min(coeffs.shape[0], n_lags)
        coeffs_row[:take] = coeffs[:take]
        rows_for_csv.append(
            [
                day_index_labels[train_start],
                day_index_labels[train_end],
                day_index_labels[test_idx],
                window_mse,
                window_mae,
                window_r2,
                aic,
                aicc,
                bic,
                *coeffs_row.tolist(),
            ]
        )

        total_sq_error += float(np.sum(residual ** 2))
        total_abs_error += float(np.sum(np.abs(residual)))
        total_count += int(y_test.shape[0])
        y_sum += float(np.sum(y_test))
        y_sum_sq += float(np.sum(y_test ** 2))
        n_windows += 1
        elapsed = time.perf_counter() - start_time
        print(f"Done in {elapsed:.3f}s, moving to next.")

    if total_count == 0:
        raise ValueError("Not enough data to evaluate the linear predictor.")

    mse = total_sq_error / total_count
    mae = total_abs_error / total_count
    mean_y = y_sum / total_count
    var_y = (y_sum_sq / total_count) - (mean_y * mean_y)
    r2 = 1.0 - (mse / var_y) if var_y > 0.0 else np.nan

    print("Linear predictor backtest results:")
    print(f"  windows={n_windows} samples={total_count} window={window} n_lags={n_lags}")
    print(f"  mse={mse:.6e} mae={mae:.6e} r2={r2:.6f}")

    save_backtest_results_csv(output_csv_path, rows_for_csv, n_lags)
    print(f"Saved evaluation details to: {output_csv_path}")

    return

if __name__ == "__main__":
    run_pipeline(
        input_data_folder=input_data_folder,
        price_truncation_mode=price_truncation_mode,
        volatility_truncation_mode=volatility_truncation_mode,
        remove_pattern=remove_pattern,
        volatility_window_size=volatility_window_size,
        normalise_average_value=normalise_average_value,
        N_autocorrelation=N_autocorrelation,
        N_years_backtest=N_years_backtest,
        output_results_csv=evaluation_output_csv,
    )
