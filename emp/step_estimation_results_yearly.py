import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from parameters import * 
from preparations import * 

from lib.volatility import * 




# Activate AV
activateAV = False
N_rolling = 4

# Initialize variables
QV_year = {year:[] for year in years}

for (year, month, day) in dates: 
    QV_year[year].append(DH.get_data(FileTypeQV(asset, year, month, day))) 

for year in years:
    QV_year[year] = np.array(QV_year[year])
    QV_year[year] = QV_year[year].sum(axis=0)




# GMM Estimation for each year
for year in years:
    H_total_id = estimation_GMM(np.identity(len(QV_year[year])),
                                QV_year[year],
                                Psi,
                                H_min,
                                H_max,
                                H_mesh)
    print(f"{year}\t{H_total_id:.4f}")

print()
print()

# GMM Estimation for each year
for i in range(len(years) - N_rolling):
    QV_total = QV_year[years[i]]
    for j in range(1, N_rolling):
        QV_total = QV_total + QV_year[years[i + j]]
    H_total_id = estimation_GMM(np.identity(len(QV_total)),
                                QV_total,
                                Psi,
                                H_min,
                                H_max,
                                H_mesh)
    print(f"{years[i]}-{years[i+N_rolling-1]}\t{H_total_id:.4f}")

print()
print()

QV_total = QV_year[years[0]]
for i in range(1, len(years)):
    QV_total = QV_total + QV_year[years[i]]
H_total_id = estimation_GMM(np.identity(len(QV_total)),
                            QV_total,
                            Psi,
                            H_min,
                            H_max,
                            H_mesh)
print(f"{years[0]}-{years[-1]}\t{H_total_id:.4f}")
