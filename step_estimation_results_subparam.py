import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from step_params import * 
from volatility import * 


# Initialise subparametrisation


# Using only lags zero and one

subparam = [
    False for x in label_array
]

subparam[0] = True
subparam[6] = True


# # Using only one window

# subparam = [
#     False for x in label_array
# ]
# s,e = 12, 18
# subparam[s:e] = [True for i in range(s, e)]

# # Using only non zero lags

# subparam = [
#     True for x in label_array
# ]
# subparam[0] = False
# subparam[12] = False
# subparam[18] = False
# subparam[22] = False





for p,l in zip(subparam, label_array):
    if p:
        print(f"Using {l}")



# subparametrisation functions

def filter_array(Y, subparam):
    """
    Filters the elements of array Y based on the boolean values in subparam.

    Parameters:
        Y (np.ndarray): The input array of size N.
        subparam (list or np.ndarray): A boolean list or array of size N with exactly m True values.

    Returns:
        np.ndarray: A new array of size m containing Y[i] where subparam[i] is True.
    """
    if len(Y) != len(subparam):
        raise ValueError("Y and subparam must have the same length.")

    # Use NumPy's boolean indexing
    return Y[np.array(subparam)]

def filter_matrix(M, subparam):
    """
    Filters the rows and columns of a matrix M based on the boolean values in subparam.

    Parameters:
        M (np.ndarray): The input matrix of size (N, N).
        subparam (list or np.ndarray): A boolean list or array of size N with exactly m True values.

    Returns:
        np.ndarray: A new matrix of size (m, m) containing M[i, j] where subparam[i] and subparam[j] are True.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")
    if M.shape[0] != len(subparam):
        raise ValueError("The size of subparam must match the dimensions of M.")

    # Use NumPy's boolean indexing to filter rows and columns
    mask = np.array(subparam)
    return M[mask][:, mask]

def compose_filter_array(fun, subparam):
    return lambda x: filter_array(fun(x), subparam)




# Initialize variables
QV = []
AV = []

for (year, month, day) in dates: 
    QV.append(DH.get_data(FileTypeQV(asset, year, month, day))) 
    AV.append(DH.get_data(FileTypeAV(asset, year, month, day))) 


# Subparametrisation 
QV = [filter_array(qv, subparam) for qv in QV]
AV = [filter_matrix(av, subparam) for av in AV]
Psi = compose_filter_array(Psi, subparam)


QV = np.array(QV)
AV = np.array(AV)




# Rolling window estimation
QV_total = QV.sum(axis=0)
QV_rolling = np.array([QV[i:i + days_estimation].sum(axis=0)
                        for i in range(len(QV) - days_estimation)]) / days_estimation

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

H_total_si = estimation_GMM(sigma_total,
                            QV_total,
                            Psi,
                            0.001,
                            0.499,
                            0.00001)

# GMM Estimation for each rolling window
estimates_H_id = []
estimates_H_si = []

for (sigma_window, QV_window) in zip(sigma_rolling, QV_rolling):
    H_id = estimation_GMM(np.identity(len(QV_window)),
                          QV_window,
                          Psi,
                          0.001,
                          0.499,
                          0.001)

    H_si = estimation_GMM(sigma_window,
                          QV_window,
                          Psi,
                          0.001,
                          0.499,
                          0.001)


    estimates_H_id.append(H_id)
    estimates_H_si.append(H_si)

# Convert estimates_H to numpy array for analysis
estimates_H_id = np.array(estimates_H_id)
estimates_H_si = np.array(estimates_H_si)

# Print results summary statistics for both sets of estimates
print(f"GMM Estimation on the entire dataset without weight matrix selection: H = {H_total_id:.5f}")
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
df_summary_si = pd.DataFrame(summary_stats_si, index=["H Estimates (SI)"])

print("Summary Statistics (ID):\n", df_summary_id)
print("\nSummary Statistics (SI):\n", df_summary_si)

# Plot the estimates
# first_days = [f"{year:04d}-{month:02d}-{day:02d}" for (year, month, day) in dates[:-days_estimation]]
first_days = [
    datetime.strptime(f"{year:04d}-{month:02d}-{day:02d}", "%Y-%m-%d")
    for (year, month, day) in dates[:-days_estimation]
]

plt.figure(figsize=(12, 8))
plt.plot(first_days, estimates_H_id, label="GMM Estimates H (ID)", color='blue', linestyle='-')
plt.plot(first_days, estimates_H_si, label="GMM Estimates H (SI)", color='red',  linestyle='-')
plt.xlabel("Date")
plt.ylabel("H Estimates")
plt.title("H Estimates Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
