import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from step_params import * 
from volatility import * 

# Initialize variables
QV = []
for (year, month, day) in dates: 
    QV.append(DH.get_data(FileTypeQV(asset, year, month, day))) 
QV = np.array(QV)

# Rolling window estimation
QV_total = QV.sum(axis=0)
QV_rolling = np.array([QV[i:i + days_estimation].sum(axis=0)
                        for i in range(len(QV) - days_estimation)]) / days_estimation
first_days = dates[:-days_estimation]

# GMM Estimation on the entire dataset (QV_total)
H_total = estimation_GMM(np.identity(len(QV_total)),
                         QV_total,
                         Psi,
                         0.001,
                         0.499,
                         0.00001)
print(f"GMM Estimation on the entire dataset (QV_total): H = {H_total}")

# GMM Estimation for each rolling window
estimates_H = []
for QV_window in QV_rolling:
    H = estimation_GMM(np.identity(len(QV_window)),
                       QV_window,
                       Psi,
                       0.001,
                       0.499,
                       0.0001)
    estimates_H.append(H)

# Convert estimates_H to numpy array for analysis
estimates_H = np.array(estimates_H)

# Plot the estimates
plt.figure(figsize=(10, 6))
plt.plot([f"{year:04d}-{month:02d}-{day:02d}" for (year, month, day) in first_days], 
         estimates_H, 
         label="GMM Estimates H", 
         color='blue')
plt.xlabel("Date")
plt.ylabel("H Estimates")
plt.title("H Estimates Over Time")
plt.legend()
plt.show()

# Summarize the main features of estimates_H
summary_stats = {
    "Mean": np.mean(estimates_H),
    "Median": np.median(estimates_H),
    "25% Quantile": np.percentile(estimates_H, 25),
    "75% Quantile": np.percentile(estimates_H, 75),
    "Min": np.min(estimates_H),
    "Max": np.max(estimates_H)
}

# Display the summary stats as a pandas DataFrame
df_summary = pd.DataFrame(summary_stats, index=["H Estimates"])
print(df_summary)
