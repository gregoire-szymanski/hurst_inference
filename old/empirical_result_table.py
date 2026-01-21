import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm

from lib.volatility import * 

from parameters import * 
from preparations import *



rolling_years = 4

H_mesh = 0.0001
H_min = H_mesh
H_max = 0.5 + H_mesh

alpha = 0.95  # Example: 95% confidence interval
z_alpha = norm.ppf((1 + alpha) / 2)  # Compute Φ^−1((1 - α) / 2)


# Initialize variables
QV_year = {year:[] for year in years}
AV_year = {year:[] for year in years}

for (year, month, day) in dates: 
    QV_year[year].append(DH.get_data(FileTypeQV(asset, year, month, day))) 
    AV_year[year].append(DH.get_data(FileTypeAV(asset, year, month, day))) 

n_days = {year: len(QV_year[year]) for year in years}

for year in years:
    QV_year[year] = np.array(QV_year[year])
    QV_year[year] = QV_year[year].mean(axis=0)

    AV_year[year] = np.array(AV_year[year])
    AV_year[year] = AV_year[year].mean(axis=0)


# GMM Estimation for each year
for i in range(len(years) - rolling_years):
    QV_total = QV_year[years[i]]
    AV_total = AV_year[years[i]]
    days_total = n_days[years[i]]

    for j in range(1, rolling_years):
        QV_total = QV_total + QV_year[years[i + j]]
        AV_total = AV_total + AV_year[years[i + j]]
        days_total = days_total + n_days[years[i + j]]
 
    QV_total = QV_total / rolling_years
    AV_total = AV_total / rolling_years

    W_total = np.linalg.inv(AV_total)

    H_estimated, R_estimated = estimation_GMM(W_total,
                                    QV_total,
                                    Psi,
                                    H_min,
                                    H_max,
                                    H_mesh)
    
    C1, C2 = get_confidence_size(params_volatility, H_estimated, R_estimated, days_total, delta, AV_total, W_total)
    
    print(f"{years[i]}-{years[i+rolling_years-1]}\t{H_estimated:.4f}\t[{H_estimated - C1 * z_alpha:.4f}, {H_estimated + C1 * z_alpha:.4f}]")

print()
print()

