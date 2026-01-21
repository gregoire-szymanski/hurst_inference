import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt

from scipy.special import gamma

# main parameters (defaults)

input_data_folder = "/Users/gregoireszymanski/Documents/data/spy/market/clean"
output_data_volume_name = "prepared_data_volume_STD_3.npy"
output_data_orders_name = "prepared_data_orders_STD_3.npy"
prefix = ""  # String prefix for data files
remove_FOMC_days = True
truncation_mode = "STD_3"
remove_trading_halts = 60

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


def max_consecutive_true(x: np.ndarray) -> int:
    """
    Return the maximum length of consecutive True in a boolean array.
    """
    if x.size == 0:
        return 0
    x = x.astype(bool)
    max_run = 0
    run = 0
    for v in x:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run



# Load FOMC dates list (YYYY-MM-DD per line)
with open(os.path.join(os.path.dirname(__file__), "../dates/FOMC.txt")) as _fomc_file:
    FOMC_dates = [line.strip() for line in _fomc_file if line.strip()]


print("Step 0/7: Checking input/output folders and creating output folder if needed...")

if input_data_folder is None:
    raise ValueError("Config error: input_data_folder is None.")

    # Step 1
print("Listing files, filtering by prefix+date format, loading prices, applying filters, and subsampling...")

# Match prefixYYYY-MM-DD.csv
# If prefix="" this expects "YYYY-MM-DD.csv"
date_re = re.compile(rf"^{re.escape(prefix)}(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")

all_files = sorted(os.listdir(input_data_folder))
csv_files = []
for fn in all_files:
    m = date_re.match(fn)
    if m:
        csv_files.append((fn, m.group(1)))

if remove_FOMC_days:
    csv_files = [(fn, d) for (fn, d) in csv_files if d not in set(FOMC_dates)]

if not csv_files:
    print("No matching CSV files found after filtering.")
    print("Finished. Summary: 0 input days processed.")
    exit()

daily_prices: List[np.ndarray] = []
daily_dates: List[str] = []

kept = 0
skipped_halts = 0
skipped_errors = 0



daily_volume: List[np.ndarray] = []
daily_orders: List[np.ndarray] = []
daily_dates: List[str] = []


method, param = parse_truncation_mode(truncation_mode)
total_orders = 0
removed_orders = 0
for fn, date_str in csv_files:
    print(fn)
    path = os.path.join(input_data_folder, fn)
    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            dtype=[("time", "U24"), ("size", "f8")],
            usecols=(1, 2),
        )
        if data.size == 0:
            daily_volume.append(np.array([], dtype=np.float64))
            daily_orders.append(np.array([], dtype=np.int64))
            daily_dates.append(date_str)
            continue

        times = data["time"]
        sizes = data["size"]
        total_orders += sizes.size

        if method == "STD":
            mean_size = sizes.mean()
            std_size = sizes.std()
            cutoff = mean_size + param * std_size
            mask = sizes <= cutoff
            removed_orders += sizes.size - int(mask.sum())
            sizes = sizes[mask]
            times = times[mask]

        if sizes.size == 0:
            daily_volume.append(np.array([], dtype=np.float64))
            daily_orders.append(np.array([], dtype=np.int64))
            daily_dates.append(date_str)
            continue

        # times is a vector of strings with format 9:30:00.012733000.
        def _seconds_from_midnight(t: str) -> int:
            h, m, rest = t.split(":")
            s = rest.split(".", 1)[0]
            return int(h) * 3600 + int(m) * 60 + int(s)

        seconds = np.fromiter((_seconds_from_midnight(t) for t in times), dtype=np.int32)
        market_open = 9 * 3600 + 30 * 60
        sec_from_open = seconds - market_open
        in_session = (sec_from_open >= 0) & (sec_from_open < 23400)
        sec_from_open = sec_from_open[in_session]
        sizes_in_session = sizes[in_session]

        volumes = np.bincount(sec_from_open, weights=sizes_in_session, minlength=23400)
        orders = np.bincount(sec_from_open, minlength=23400)

        if remove_trading_halts is not None and int(remove_trading_halts) >= 0:
            halts = (volumes == 0)
            max_run = max_consecutive_true(halts)
            if max_run > int(remove_trading_halts):
                skipped_halts += 1
                continue


        daily_volume.append(volumes.astype(np.float64, copy=False))
        daily_orders.append(orders.astype(np.int64, copy=False))
        daily_dates.append(date_str)
        kept += 1

    except Exception:
        skipped_errors += 1

if total_orders > 0:
    removed_ratio = removed_orders / total_orders
else:
    removed_ratio = 0.0
print(f"Removed orders: {removed_orders} ({removed_ratio:.2%} of {total_orders})")

daily_volume_array = np.array(daily_volume, dtype=object)  # Object array of 1D arrays
daily_orders_array = np.array(daily_orders, dtype=object)  # Object array of 1D arrays

np.save(output_data_volume_name, daily_volume_array)
np.save(output_data_orders_name, daily_orders_array)
