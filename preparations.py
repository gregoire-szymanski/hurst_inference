import os
from datetime import datetime

import numpy as np

from lib.estimator_H import *
from lib.data_handler import DataHandler, FileType
from lib.dates import *

from parameters import *



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
DH = DataHandler(prices_folder="~/Documents/data/" + asset + "/price/1s/daily_csv/", 
                 tmp_folder=tmp_folder)




# Kernel for W_fun
if W_fun_id == 'parzen':
    kernel_k = lambda x: 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x)**3
    W_fun = lambda Lmax, L: kernel_k(np.abs(L / Lmax))
else:
    raise ValueError("Unknown W_fun_id")


# FileType generators
def FileTypeVolatility(asset, year, month, day, window):
    subfolder = f"vol/{window}"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

def FileTypePattern(asset, window):
    subfolder = f"pattern/{window}"
    return FileType(subfolder=subfolder, asset=asset)

def FileTypeVolatilityIncrements(asset, year, month, day, window):
    subfolder = f"volinc/{window}"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

def FileTypeQV(asset, year, month, day):
    subfolder = "qv/"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)

def FileTypeAV(asset, year, month, day):
    subfolder = "av/"
    return FileType(subfolder=subfolder, asset=asset, year=year, month=month, day=day)


# Remove FOMC and trading halts
for date in FOMC_announcement:
    DH.remove_date(date)

for date in trading_halt:
    DH.remove_date(date)

dates = DH.dates(asset)
dates = [(date.year, date.month, date.day) for D in dates for date in [datetime.strptime(D, "%Y-%m-%d")]]

years = [year for (year, month, day) in dates ]
years = np.unique(years)






# Create estimation functions
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
    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']

        # factor = window^(2H)
        factor = window**(2 * H)
        
        # Compute the first two terms outside the loop
        p.append(factor * (Phi_Hl(0, H) + 2 * Phi_Hl(1, H)))

        # Compute remaining terms for i in [2, N_lags]
        for i in range(2, N_lags + 1):
            p.append(factor * Phi_Hl(i, H))
    return np.array(p)


# Params array

window_array = []
lags_array = []
label_array = []

for param in params_volatility:
    window = param['window']
    N_lags = param['N_lags']
    
    # Extend window_array with repeated window entries
    window_array.extend([window for _ in range(N_lags)])
    lags_array.append("Lag_0 + 2 * Lag_1")
    lags_array.extend([f"Lag_{i}" for i in range(2,N_lags+1)])
    label_array.append(f"W{window}; L0 + 2 L1")
    label_array.extend([f"W{window}; L{i}" for i in range(2,N_lags+1)])


# Data Loader
def load_QV():
    QV = []
    for (year, month, day) in dates: 
        QV.append(DH.get_data(FileTypeQV(asset, year, month, day))) 
    QV = np.array(QV)
    return QV

def load_AV():
    AV = []
    for (year, month, day) in dates: 
        AV.append(DH.get_data(FileTypeAV(asset, year, month, day))) 
    AV = np.array(AV)
    return AV
