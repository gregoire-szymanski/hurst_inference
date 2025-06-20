import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

print("Estimated Asymptotic variance:")
print(AV_total)
print()

print("Estimated H:", H_total)
print("Estimated R:", R_total)

print()

print("Real Asymptotic variance:")
print(get_theoretical_variance(params_volatility, H_total))

# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot first heatmap
sns.heatmap(AV_total, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes[0])
axes[0].set_title("Correlation Matrix Heatmap (AV_total)")
axes[0].set_xlabel("Variables")
axes[0].set_ylabel("Variables")

# Plot second heatmap
sns.heatmap(get_theoretical_variance(params_volatility, H_total), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes[1])
axes[1].set_title("Correlation Matrix Heatmap (Theoretical Variance)")
axes[1].set_xlabel("Variables")
axes[1].set_ylabel("Variables")

# Show the figure
plt.tight_layout()
plt.show()


print("Ratio Asymptotic variance:")
print(AV_total / get_theoretical_variance(params_volatility, H_total))


print(C1 / np.sqrt(len(QV)))

exit()

# GMM Estimation for each rolling window
estimates_H = []

for (sigma_window, QV_window) in zip(W_rolling, QV_rolling):
    H_si, _ = estimation_GMM(sigma_window,
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
