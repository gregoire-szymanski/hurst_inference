import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lib.volatility import * 

from parameters import * 
from preparations import *

# Activate AV
days_estimation = 252*2
H_mesh = 0.01
H_min = H_mesh
H_max = 0.5 + H_mesh


# Load data
QV = load_QV()
AV = load_AV()

# Rolling window estimation
QV_total = QV.sum(axis=0)
QV_rolling = np.array([QV[i:i + days_estimation].sum(axis=0)
                        for i in range(len(QV) - days_estimation)]) / days_estimation

AV_total = AV.sum(axis=0)
AV_rolling = np.array([AV[i:i + days_estimation].sum(axis=0)
                            for i in range(len(QV) - days_estimation)]) / days_estimation

sigma_total = np.linalg.inv(AV_total)
sigma_rolling = np.array([np.linalg.inv(av) for av in AV_rolling])


H_total = estimation_GMM(sigma_total,
                            QV_total,
                            Psi,
                            H_min,
                            H_max,
                            H_mesh)

# GMM Estimation for each rolling window
estimates_H = []

for (sigma_window, QV_window) in zip(sigma_rolling, QV_rolling):
    H_si = estimation_GMM(sigma_window,
                              QV_window,
                              Psi,
                              H_min,
                              H_max,
                              H_mesh)


    estimates_H.append(H_si)

# Convert estimates_H to numpy array for analysis
estimates_H = np.array(estimates_H)

# Print results summary statistics for both sets of estimates
print(f"GMM Estimation on the entire dataset : H = {H_total:.5f}")

# Output summary statistics for both sets of estimates
summary_stats = {
    "Mean": np.mean(estimates_H),
    "Median": np.median(estimates_H),
    "25% Quantile": np.percentile(estimates_H, 25),
    "75% Quantile": np.percentile(estimates_H, 75),
    "Min": np.min(estimates_H),
    "Max": np.max(estimates_H)
}

# Display the summary stats as pandas DataFrames
df_summary = pd.DataFrame(summary_stats, index=["H Estimates"])

print("Summary Statistics:\n", df_summary)

# Plot the estimates
# first_days = [f"{year:04d}-{month:02d}-{day:02d}" for (year, month, day) in dates[:-days_estimation]]
first_days = [
    datetime.strptime(f"{year:04d}-{month:02d}-{day:02d}", "%Y-%m-%d")
    for (year, month, day) in dates[:-days_estimation]
]

plt.figure(figsize=(12, 8))
plt.plot(first_days, estimates_H, color='blue', linestyle='-')
plt.xlabel("$t$")
plt.ylabel("$H$")
# plt.title("H Estimates Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
