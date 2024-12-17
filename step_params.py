import numpy as np
import os
from datetime import datetime
from data_handler import DataHandler, FileType
from dates import *


print("Initialisation...")


# Identification
identificator = "test_1s"

# Global parameters
asset = 'spy'
subsampling = 1
delta = 1.0 / (252.0 * 23400) * subsampling  # Time increment
days_estimation = 252

# Volatility and quadratic variation estimation parameters
price_truncation_method = 'BIVAR3'
vol_truncation_method = 'STD3'
params_volatility = [
    {'window': 150, 'N_lags': 12},
    {'window': 300, 'N_lags': 6},
    {'window': 450, 'N_lags': 4},
    {'window': 600, 'N_lags': 3},
]

# Asymptotic variance estimation
Ln = 1800
Kn = 900
W_fun_id = 'parzen'



#### DO NOT TOUCH BELOW

# Create folder structure
tmp_folder = os.path.expanduser(f"~/Documents/data/tmp/hurst_inference/{identificator}/")

# Function to create folders dynamically
def create_folders(base_folder, subfolders):
    for folder in subfolders:
        os.makedirs(os.path.join(base_folder, folder), exist_ok=True)

# Main folders
create_folders(tmp_folder, ['vol', 'pattern', "volinc", 'qv', 'av', "result"])
create_folders(tmp_folder, [f"vol/{param['window']}" for param in params_volatility])
create_folders(tmp_folder, [f"pattern/{param['window']}" for param in params_volatility])
create_folders(tmp_folder, [f"volinc/{param['window']}" for param in params_volatility])

# Create date handler
DH = DataHandler(prices_folder="~/Documents/data/SPY/price/1s/daily_csv/", 
                 tmp_folder=tmp_folder)

# Kernel for W_fun
if W_fun_id == 'parzen':
    kernel_k = lambda x: 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x)**3
    W_fun = lambda Lmax, L: kernel_k(np.abs(L / Lmax))
else:
    raise ValueError("Unknown W_fun_id")

# Function to extract date and asset from file name
# def parse_price_file(priceFile):
#     """
#     Extract asset, year, month, and day from price file in the format 'xxx_YYYY-MM-DD.csv'.
#     """
#     basename = os.path.basename(priceFile)
#     asset, date_str = basename.split('_')
#     date = datetime.strptime(date_str.replace('.csv', ''), "%Y-%m-%d")
#     return asset, date.year, date.month, date.day

# FileType generators
def FileTypeVolatility(asset, year, month, day, window):
    # asset, year, month, day = parse_price_file(priceFile)
    subfolder = f"vol/{window}"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

def FileTypePattern(asset, window):
    # asset, year, month, day = parse_price_file(priceFile)
    subfolder = f"pattern/{window}"
    return FileType(subfolder=subfolder, asset=asset)

def FileTypeVolatilityIncrements(asset, year, month, day, window):
    # asset, year, month, day = parse_price_file(priceFile)
    subfolder = f"volinc/{window}"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

def FileTypeQV(asset, year, month, day):
    # asset, year, month, day = parse_price_file(priceFile)
    subfolder = "qv/"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

def FileTypeAV(asset, year, month, day, window, N_lags):
    # asset, year, month, day = parse_price_file(priceFile)
    subfolder = "av/"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

# Remove all FOMC announcement dates from the data
for date in FOMC_announcement:
    DH.remove_date(date)

# Remove all trading halt dates from the data
for date in trading_halt:
    DH.remove_date(date)

dates = DH.dates(asset)
dates = [(date.year, date.month, date.day) for D in dates for date in [datetime.strptime(D, "%Y-%m-%d")]]