import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from step_params import * 
from volatility import * 

# Activate AV
activateAV = True

# Initialize variables
vols = []
for (year, month, day) in dates: 
    vols.append(np.mean([
        np.mean(DH.get_data(FileTypeVolatility(asset, year, month, day, param['window'])))
        for param in params_volatility
    ])) 
vols = np.array(vols)

QV = []
for (year, month, day) in dates: 
    QV.append(DH.get_data(FileTypeQV(asset, year, month, day))) 
QV = np.array(QV)

if activateAV:
    AV = []
    for (year, month, day) in dates: 
        AV.append(DH.get_data(FileTypeAV(asset, year, month, day))) 
    AV = np.array(AV)



# Rolling window estimation
QV_total = QV.sum(axis=0)
QV_rolling = np.array([QV[i:i + days_estimation].sum(axis=0)
                        for i in range(len(QV) - days_estimation)]) / days_estimation

if activateAV:
    AV_total = AV.sum(axis=0)
    AV_rolling = np.array([AV[i:i + days_estimation].sum(axis=0)
                            for i in range(len(QV) - days_estimation)]) / days_estimation

    sigma_total = np.linalg.inv(AV_total)
    sigma_rolling = np.array([np.linalg.inv(av) for av in AV_rolling])

vol_rolling = np.array([vols[i:i + days_estimation].sum(axis=0)
                        for i in range(len(QV) - days_estimation)]) / days_estimation

# GMM Estimation for each rolling window
estimates_H_id = []
estimates_H_si = []

for  QV_window in QV_rolling:
    H_id = estimation_GMM(np.identity(len(QV_window)),
                          QV_window,
                          Psi,
                          0.001,
                          0.499,
                          0.001)
    estimates_H_id.append(H_id)

if activateAV:
    for (sigma_window, QV_window) in zip(sigma_rolling, QV_rolling):
        H_si = estimation_GMM(sigma_window,
                            QV_window,
                            Psi,
                            0.001,
                            0.499,
                            0.001)


        estimates_H_si.append(H_si)

# Convert estimates_H to numpy array for analysis
estimates_H_id = np.array(estimates_H_id)
estimates_H_si = np.array(estimates_H_si)

# Output summary statistics for both sets of estimates
summary_stats_id = {
    "Mean": np.mean(estimates_H_id),
    "Median": np.median(estimates_H_id),
    "25% Quantile": np.percentile(estimates_H_id, 25),
    "75% Quantile": np.percentile(estimates_H_id, 75),
    "Min": np.min(estimates_H_id),
    "Max": np.max(estimates_H_id)
}

if activateAV:
    summary_stats_si = {
        "Mean": np.mean(estimates_H_si),
        "Median": np.median(estimates_H_si),
        "25% Quantile": np.percentile(estimates_H_si, 25),
        "75% Quantile": np.percentile(estimates_H_si, 75),
        "Min": np.min(estimates_H_si),
        "Max": np.max(estimates_H_si)
    }

# Display the summary stats as pandas DataFrames
df_summary_id = pd.DataFrame(summary_stats_id, index=["H Estimates (ID)"])
if activateAV:
    df_summary_si = pd.DataFrame(summary_stats_si, index=["H Estimates (SI)"])

print("Summary Statistics (ID):\n", df_summary_id)
if activateAV:
    print("\nSummary Statistics (SI):\n", df_summary_si)

# Plot the estimates
# first_days = [f"{year:04d}-{month:02d}-{day:02d}" for (year, month, day) in dates[:-days_estimation]]
first_days = [
    datetime.strptime(f"{year:04d}-{month:02d}-{day:02d}", "%Y-%m-%d")
    for (year, month, day) in dates[:-days_estimation]
]

fig, ax1 = plt.subplots(figsize=(12, 8))

# First y-axis: GMM Estimates H (ID) and (SI)
ax1.plot(first_days, estimates_H_id, label="GMM Estimates H (ID)", color='blue', linestyle='-')
if activateAV:
    ax1.plot(first_days, estimates_H_si, label="GMM Estimates H (SI)", color='red', linestyle='-')

ax1.set_xlabel("Date")
ax1.set_ylabel("H Estimates")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("H Estimates and Volatility Over Time")
ax1.legend(loc='upper left')

# Second y-axis: Volatility
ax2 = ax1.twinx()
ax2.plot(first_days, vol_rolling, label="Rolling Volatility", color='green', linestyle='--')
ax2.set_ylabel("Volatility")
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

# Final adjustments
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Compute and print covariance between estimates_H_id and vol_rolling
cor_id_vol = np.corrcoef(estimates_H_id.flatten(), vol_rolling.flatten())[0, 1]
print(f"Correlation between H Estimates (ID) and Rolling Volatility: {cor_id_vol:.4f}")

if activateAV:
    # Compute and print covariance between estimates_H_si and vol_rolling
    cor_si_vol = np.corrcoef(estimates_H_si.flatten(), vol_rolling.flatten())[0, 1]
    print(f"Correlation between H Estimates (SI) and Rolling Volatility: {cor_si_vol:.4f}")

