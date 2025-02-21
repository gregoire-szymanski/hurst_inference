from step_params import *
from volatility import *
from timer import *
from quadratic_variation import *


for (i,param) in enumerate(params_volatility):
    param["pattern"] = DH.get_data(FileTypePattern(asset, param["window"]))
    param["qve"] = QuadraticCovariationsEstimator(param["window"], 
                                                  param["N_lags"] + 1, 
                                                  vol_truncation_method)

   
ave = AsymptoticVarianceEstimator(W_fun, Ln, Kn)



print("Computing asymptotic variance...")
timer = Timer(len(dates), type="window")
timer.start()
for (i,(year, month, day)) in enumerate(dates):
    if i % 20 == 19 and i > 0:  timer.step(i)
    #timer.step(i)   

    sigma = np.zeros((len(window_array), len(window_array)))

    # Compute daily Realized Variation (DRV) for each param set
    DRV = []
    for param in params_volatility:
        # Extract objects
        qve = param["qve"] 
        pattern = param["pattern"]
        
        vol = DH.get_data(FileTypeVolatility(asset, year, month, day, param["window"]))
        volinc = param["qve"].precompute(vol, param["pattern"])

        # Compute DRV
        DRV.extend(qve.DRV(volinc))


    # Apply correction and compute sum_Sigma terms
    psi = [ave.correction(drv) for drv in DRV]
        
    for idx_i in range(len(window_array)):
        for idx_j in range(len(window_array)):
            sigma[idx_i, idx_j] += ave.compute_pos(
                psi[idx_i], psi[idx_j], 
                window_array[idx_i], window_array[idx_j]
            )
    sigma = sigma + sigma.transpose()

    for idx_i in range(len(window_array)):
        for idx_j in range(len(window_array)):
            sigma[idx_i, idx_j] += ave.compute0(
                psi[idx_i], psi[idx_j], 
                window_array[idx_i], window_array[idx_j]
            )

    DH.save_data(FileTypeAV(asset, year, month, day), 
                 sigma)

    
print(f"Asymptotic variances computed in {timer.total_time():.2f}s.")




