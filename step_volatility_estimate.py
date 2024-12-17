from step_params import *
from volatility import *
from timer import *

for param in params_volatility:
    window = param['window']
    N_lags = param['N_lags']

    # Initialize volatility and quadratic variation estimators
    param["ve"] = VolatilityEstimator(
        delta=delta,
        window=window,
        price_truncation=price_truncation_method
    )


print("Computing intraday volatilities...")
timer = Timer(len(dates))
timer.start()
for (i,(year, month, day)) in enumerate(dates):
    if i % 50 == 49 and i > 0:  timer.step(i)

    price = DH.get_price(asset, year, month, day)
    price.subsample(subsampling)
    price_array = price.get_price() 

    for param in params_volatility:
        vol = param["ve"].compute(price_array)
        filetype = FileTypeVolatility(asset, year, month, day, param["window"])
        DH.save_data(filetype, vol.get_values())
print(f"Intraday volatilities computed in {timer.total_time():.2f}s.")




