import numpy as np
import matplotlib.pyplot as plt

from data_handler import DataHandler

# Initialize the data handler
DH = DataHandler(prices_folder="~/Documents/data/SPY/price/1s/daily_csv/", 
                tmp_folder="~/Documents/data/tmp/hurst_inference")


max_window = 300
eps = 0.001

trading_halt = []

for price_file in DH.price_files:
    price = DH.get_price(price_file).get_price()
    increment = np.abs(price[1:] - price[:-1]) < eps
    count = 0
    i = 0
    if len(price) < 23400:
        print(f"Short trading day detected in file {price_file}")
        trading_halt.append(price_file)

    else:
        while i < len(increment) and count < max_window:
            if increment[i]:
                count += 1
            else:
                count = 0
            i += 1
        if count == max_window:
            print(f"Trading halt detected in file {price_file}")
            trading_halt.append(price_file)
            if len(trading_halt) == 2:
                plt.plot(price)

print(f"trading_halt = {trading_halt}")
plt.show()