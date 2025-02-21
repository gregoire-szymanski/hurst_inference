from step_params import *
from volatility import *
from timer import *
from quadratic_variation import *

print("Computing volatility increments...")
timer = Timer(len(params_volatility), type="window")
timer.start()
for (i,param) in enumerate(params_volatility):
    timer.step(i)

    pattern = DH.get_data(FileTypePattern(asset, param["window"]))

    for (year, month, day) in dates:
        vol = DH.get_data(FileTypeVolatility(asset, year, month, day, param["window"]))
        volinc = computeVolatilityIncrements(param["window"], vol_truncation_method, vol, pattern)
        DH.save_data(FileTypeVolatilityIncrements(asset, year, month, day, param["window"]), 
                     volinc)

    
print(f"Volatility increments computed in {timer.total_time():.2f}s.")




