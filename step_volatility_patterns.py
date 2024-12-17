from step_params import *
from volatility import *
from timer import *


print("Computing volatility patterns...")
timer = Timer(len(params_volatility), type="window")
timer.start()
for (i,param) in enumerate(params_volatility):
    timer.step(i)
    pattern = VolatilityPattern()

    for (year, month, day) in dates:
        filetype = FileTypeVolatility(asset, year, month, day, param["window"])
        vol = DH.get_data(filetype)
        pattern.accumulate(vol)
    
    filetype = FileTypePattern(asset, param["window"])
    DH.save_data(filetype, pattern.get_pattern().get_values())

print(f"Volatility patterns computed in {timer.total_time():.2f}s.")




