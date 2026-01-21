from lib.price import Price
from lib.data_handler import DataHandler
from datetime import datetime
import os
import numpy as np


tmp_folder = os.path.expanduser(f"~/Documents/data/tmp/hurst_inference/dummy/")

DH = DataHandler(prices_folder="~/Documents/data/SPY/price/1s/daily_csv/", 
                 tmp_folder=tmp_folder)

asset = 'spy'
dates = DH.dates(asset)
dates = [(date.year, date.month, date.day) for D in dates for date in [datetime.strptime(D, "%Y-%m-%d")]]

first_price = []
last_price = []

for (i,(year, month, day)) in enumerate(dates):
    price = Price(DH.get_price(asset, year, month, day))
    price = price.get_price()
    first_price.append(price[0])
    last_price.append(price[-1])
    
first_price = np.array(first_price)
last_price = np.array(last_price)

intraday_profit = last_price - first_price
overnight_profit = first_price[1:] - last_price[:-1]


cash = 1
for i in range(len(dates)):
    long = cash / first_price[i]
    cash = long * last_price[i]
print("intraday trading: ", cash)

cash = 1
for i in range(len(dates) - 1):
    long = cash / last_price[i]
    cash = long * first_price[i+1]
print("overnight trading: ", cash)






