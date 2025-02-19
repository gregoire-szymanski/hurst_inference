from step_params import *
from volatility import *
from timer import *
import matplotlib.pyplot as plt


print("Computing volatility patterns...")
timer = Timer(len(params_volatility), type="window")
timer.start()
for (i,param) in enumerate(params_volatility):
    timer.step(i)
    pattern = VolatilityPattern()

    for (year, month, day) in dates:
        vol = DH.get_data(FileTypeVolatility(asset, year, month, day, param["window"]))
        pattern.accumulate(vol)
    
    DH.save_data(FileTypePattern(asset, param["window"]), 
                 pattern.get_pattern().get_values())
    
    plt.show()

print(f"Volatility patterns computed in {timer.total_time():.2f}s.")




