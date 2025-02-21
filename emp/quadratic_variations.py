from lib.quadratic_variation import *

from lib.volatility import *
from lib.timer import *

from emp.parameters import *
from emp.preparations import *

for (i,param) in enumerate(params_volatility):
    param["pattern"] = DH.get_data(FileTypePattern(asset, param["window"]))
    param["qve"] = QuadraticCovariationsEstimator(param["window"], 
                                                  param["N_lags"] + 1, 
                                                  vol_truncation_method)



print("Computing quadratic variations...")
timer = Timer(len(dates), type="window")
timer.start()
for (i,(year, month, day)) in enumerate(dates):
    if i % 50 == 49 and i > 0:  timer.step(i)
    QV = []
    for param in params_volatility:
        vol = DH.get_data(FileTypeVolatility(asset, year, month, day, param["window"]))
        volinc = param["qve"].precompute(vol, param["pattern"])
        qv = param["qve"].conclude(volinc)
        QV.extend(qv)
    DH.save_data(FileTypeQV(asset, year, month, day), 
                 QV)

print(f"Quadratic variations computed in {timer.total_time():.2f}s.")




