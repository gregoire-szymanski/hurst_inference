from step_params import *
from volatility import *
import matplotlib.pyplot as plt

QV = []
for (year, month, day) in dates:
    QV.append(DH.get_data(FileTypeQV(asset, year, month, day)))
QV = np.array(QV)

QV_rolling = np.array([QV[i:i+days_estimation].sum(axis=0) for i in range(len(QV) - days_estimation)]) / days_estimation
first_days = dates[:-days_estimation]

estimates_H = []
for QV in QV_rolling:

    H = estimation_GMM(np.identity(len(QV)),
                       QV,
                       Psi,
                       0.001,
                       0.499,
                       0.001)
    
    estimates_H.append(H)

plt.plot(estimates_H)
plt.show()


