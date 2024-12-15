import numpy as np
import matplotlib.pyplot as plt
import random

from data_handler import DataHandler
from dates import FOMC_announcement, trading_halt
from price import Price
from volatility import VolatilityEstimator, volatility_pattern, bipower_average_V, Volatility
from quadratic_variation import QuadraticCovariationsEstimator
from estimator_H import estimation_01_2


# Initialize the data handler
DH = DataHandler(prices_folder="~/Documents/data/SPY/price/1s/daily_csv/", 
                tmp_folder="~/Documents/data/tmp/hurst_inference")

# Remove all FOMC announcement dates from the data
for date in FOMC_announcement:
    DH.remove_date(date)

# Remove all trading halt dates from the data
for date in trading_halt:
    DH.remove_date(date)

asset = 'spy'
subsampling = 5
window = 60
price_truncation_method = 'BIVAR3'
vol_truncation_method = 'STD3'
delta = 1.0 / (252.0 * 23400) * subsampling

# Construct the volatility estimator
ve = VolatilityEstimator(delta=delta, 
                         window=window, 
                         price_truncation=price_truncation_method)

qve = QuadraticCovariationsEstimator(window=window, 
                                     N_lags=10, 
                                     vol_truncation=vol_truncation_method)

all_price_files = [f for f in DH.price_files if f.startswith(asset+'_')]
all_dates = [f.split('_')[1].replace('.csv','') for f in all_price_files]


all_volatilities = []
for d in all_dates:
    y, m, day = map(int, d.split('-'))
    # Get price data as a DataFrame
    price = DH.get_price(asset, y, m, day)
    price.subsample(subsampling)
    price_array = price.get_price()  # numpy array of prices

    # Compute volatility using VolatilityEstimator
    vol = ve.compute(price_array)  
    all_volatilities.append(vol)


full_pattern = volatility_pattern(all_volatilities)




QV01 = []
QV2 = []

for vol in all_volatilities:
    tmp = qve.compute(volatilities=vol.get_values(), pattern=full_pattern.get_values())
    QV01.append(tmp[0] + 2 * tmp[1])
    QV2.append(tmp[2])

QV01 = np.mean(QV01)
QV2 = np.mean(QV2)

print(f"{QV01:.4f}\t{QV2:.4f}\t{estimation_01_2(QV01, QV2):.2f}")


