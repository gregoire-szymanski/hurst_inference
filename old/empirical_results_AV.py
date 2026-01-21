import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm

from lib.volatility import * 

from parameters import * 
from preparations import *

# Activate AV
days_estimation = 252*4
H_mesh = 0.0001
H_min = H_mesh
H_max = 0.5 + H_mesh

alpha = 0.95  # Example: 95% confidence interval
z_alpha = norm.ppf((1 + alpha) / 2)  # Compute Φ^−1((1 - α) / 2)

# Load data
QV = load_QV()
AV = load_AV()

# Rolling window estimation
QV_total = QV.mean(axis=0) 
QV_rolling = np.array([QV[i:i + days_estimation].sum(axis=0)
                        for i in range(len(QV) - days_estimation)]) / days_estimation

AV_total = AV.mean(axis=0)
AV_rolling = np.array([AV[i:i + days_estimation].sum(axis=0)
                            for i in range(len(QV) - days_estimation)]) / days_estimation

W_total = np.linalg.inv(AV_total)
W_rolling = np.array([np.linalg.inv(av) for av in AV_rolling])


H_total, R_total = estimation_GMM(W_total,
                            QV_total,
                            Psi,
                            H_min,
                            H_max,
                            H_mesh)

C1, C2 = get_confidence_size(params_volatility, H_total, R_total, len(QV), delta, AV_total, W_total)

print(f"Hurst index: {H_total:.4f}")
print(f"Confidence interval size: {C1 * z_alpha:.4f}")
print(f"Confidence interval: [{H_total - C1 * z_alpha:.4f}, {H_total + C1 * z_alpha:.4f}]")

# GMM Estimation for each rolling window
estimates_H = []
confidence_band = []

for (sigma_window, QV_window) in zip(W_rolling, QV_rolling):
    H_estimated, R_estimated = estimation_GMM(sigma_window,
                                              QV_window,
                                              Psi,
                                              H_min,
                                              H_max,
                                              H_mesh)
    
    C1, C2 = get_confidence_size(params_volatility, H_estimated, R_estimated, days_estimation, delta, AV_total, W_total)

    estimates_H.append(H_estimated)
    confidence_band.append(C1 * z_alpha)

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
    "Max": np.max(estimates_H),
    "Mean confidence": np.mean(confidence_band),

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
plt.fill_between(first_days, estimates_H - confidence_band, estimates_H + confidence_band, 
                 color='blue', alpha=0.2, label=f"{alpha*100:.0f}% Confidence Band")
plt.xlabel("$t$")
plt.ylabel("$H$")
# plt.title("H Estimates Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
