import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt

from scipy.special import gamma

# main parameters (defaults)

input_data_folder = "/Users/gregoireszymanski/Documents/data/spy/price/1s/daily_csv"
output_data_name = "prepared_data.npy"
prefix = "spy_"  # String prefix for data files
remove_FOMC_days = True
subsampling = 5 
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


def try_load_price(csv_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load CSV and return (price_array, halts_boolean_array_or_None).

    - Price column: "price" or "close" (case-insensitive).
    """
    import csv

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        field_map = {name.lower().strip(): name for name in reader.fieldnames}

        price_key = None
        for k in ("price", "close"):
            if k in field_map:
                price_key = field_map[k]
                break
        if price_key is None:
            raise ValueError(f"No 'price' or 'close' column found in {csv_path}")

        prices: List[float] = []

        for row in reader:
            raw_p = row.get(price_key, "")
            if raw_p is None or str(raw_p).strip() == "":
                continue
            try:
                p = float(raw_p)
            except ValueError:
                continue
            prices.append(p)

    price_arr = np.asarray(prices, dtype=float)

    return price_arr


# Load FOMC dates list (YYYY-MM-DD per line)
with open(os.path.join(os.path.dirname(__file__), "dates", "../FOMC.txt")) as _fomc_file:
    FOMC_dates = [line.strip() for line in _fomc_file if line.strip()]


print("Step 0/7: Checking input/output folders and creating output folder if needed...")

if input_data_folder is None:
    raise ValueError("Config error: input_data_folder is None.")

    # Step 1
# print("Step 1/7: Listing files, filtering by prefix+date format, loading prices, applying filters, and subsampling...")
# X = np.load("prepared_data.npy", allow_pickle=True)
# print(X.shape)

print("Step 1/7: Listing files, filtering by prefix+date format, loading prices, applying filters, and subsampling...")

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

for fn, date_str in csv_files:
    path = os.path.join(input_data_folder, fn)
    try:
        prices = try_load_price(path)
        if prices.size == 0:
            skipped_errors += 1
            continue

        # Remove day if it contains a trading halt run longer than remove_trading_halts periods
        if remove_trading_halts is not None and int(remove_trading_halts) >= 0:
            halts = (prices[1:] == prices[:-1])
            max_run = max_consecutive_true(halts)
            if max_run > int(remove_trading_halts):
                skipped_halts += 1
                continue

        # Subsampling
        if subsampling is not None and int(subsampling) > 1:
            prices = prices[::int(subsampling)]

        # Need enough points for window

        daily_prices.append(prices)
        daily_dates.append(date_str)
        kept += 1
    except Exception:
        skipped_errors += 1
        continue

if kept == 0:
    print("No usable days after loading/filtering.")
    print(f"Finished. Summary: files={len(csv_files)}, kept=0, skipped_halts={skipped_halts}, skipped_errors={skipped_errors}")


daily_prices_array = np.array(daily_prices, dtype=object)  # Object array of 1D arrays
np.save(output_data_name, daily_prices_array)