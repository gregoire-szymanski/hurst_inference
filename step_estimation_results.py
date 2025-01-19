import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from step_params import * 
from volatility import * 

# Activate AV
activateAV = False

# Initialize variables
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

for lab, qv in zip(label_array, QV_total):
    print(f"{lab}\t{qv:.3}")

if activateAV:
    AV_total = AV.sum(axis=0)
    AV_rolling = np.array([AV[i:i + days_estimation].sum(axis=0)
                            for i in range(len(QV) - days_estimation)]) / days_estimation

    sigma_total = np.linalg.inv(AV_total)
    sigma_rolling = np.array([np.linalg.inv(av) for av in AV_rolling])


# GMM Estimation on the entire dataset (QV_total)
H_total_id = estimation_GMM(np.identity(len(QV_total)),
                            QV_total,
                            Psi,
                            0.001,
                            0.499,
                            0.00001)

if activateAV:
    H_total_si = estimation_GMM(sigma_total,
                                QV_total,
                                Psi,
                                0.001,
                                0.499,
                                0.00001)

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

# Print results summary statistics for both sets of estimates
print(f"GMM Estimation on the entire dataset without weight matrix selection: H = {H_total_id:.5f}")
if activateAV:
    print(f"GMM Estimation on the entire dataset with optimal matrix selection:   H = {H_total_si:.5f}")

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

plt.figure(figsize=(12, 8))
plt.plot(first_days, estimates_H_id, label="GMM Estimates H (ID)", color='blue', linestyle='-')
if activateAV:
    plt.plot(first_days, estimates_H_si, label="GMM Estimates H (SI)", color='red',  linestyle='-')
plt.xlabel("Date")
plt.ylabel("H Estimates")
plt.title("H Estimates Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
