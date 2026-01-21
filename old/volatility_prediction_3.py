from lib.timer import *
from lib.volatility import *

from parameters import *
from preparations import *

from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

H_input = 0.25

H_values = np.linspace(0.01, 0.5, 50)
H_values = np.linspace(-0.495, 0.495, 100)
res_means = []
res_medians = []
#H_values = [-0.2, 0.2]


for H_input in H_values:
    print(f"H:        {H_input:.2e}")
    for param in params_volatility:
        window = param["window"]
        lags = param["N_lags"]

        param["pattern"] = DH.get_data(FileTypePattern(asset, param["window"]))


        first_col = [Phi_Hl(k, H_input) for k in range(0, lags - 1)]
        Gamma = toeplitz(first_col)
        
        # 2) Build the right‑hand side γ vector: γ[j] = φ_HL(j+1, H)
        gamma = np.array([Phi_Hl(j + 1, H_input) + 2 * Phi_Hl(j, H_input) for j in range(1, lags)])
        #print(gamma)
        
        # 3) Solve Γ·c = γ rather than inverting Γ
        try:
            coefs = np.linalg.solve(Gamma, gamma)
        except np.linalg.LinAlgError:
            # if Γ is singular, add a tiny ridge for stability
            eps = 1e-8 * np.trace(Gamma)
            coefs = np.linalg.solve(Gamma + eps * np.eye(lags), gamma)
        
        # 4) Store results back in the dict
        param["Gamma"] = Gamma
        param["gamma"] = gamma
        param["coefs"] = coefs
        
        print("window =", param["window"], "→ coefs =", coefs)


    
    

    timer = Timer(len(params_volatility), type="window")
    timer.start()



    for param in params_volatility:
        all_errors = []

        window = param["window"]
        lags = param["N_lags"]

        for (i,(year, month, day)) in enumerate(dates):
            vol = DH.get_data(FileTypeVolatility(asset, year, month, day, param["window"]))
            vol = vol / param["pattern"]
            volinc = vol[window:] - vol[:-window]
            std = volinc.std()
            volinc[np.abs(volinc) > std] = 0
            size = volinc.size
            new_size = size - (lags + 1) * window
            y_0 = volinc[(lags + 1)*window:]
            y_1 = volinc[lags*window: size - window]
            y = y_0 + 2 * y_1

            X = [
                volinc[(lags + 1 - i)*window: size - i*window]
                for i in range(2, lags + 1)
            ]

            X_mat = np.column_stack(X)       # or: np.vstack(X).T

            # 1) make the prediction via a dot‑product
            y_star = X_mat.dot(param["coefs"])

            # 2) compute the mean squared error
            error = np.mean((y - y_star)**2)

            all_errors.append(error)

            
            if False:
                # start fresh figure
                fig, ax = plt.subplots(figsize=(8, 4))

                # plot today vs first and last lag
                ax.plot(y,        label="y (t)",      linestyle="-",  alpha=0.9)
                ax.plot(X[0],     label="lag 1",      linestyle="-", alpha=0.7)
                ax.plot(X[-1],    label=f"lag {lags}", linestyle="-",  alpha=0.7)

                # labels, title, legend, grid
                ax.set_xlabel("Time index")
                ax.set_ylabel(f"Vol increment (window={window})")
                ax.set_title(f"{asset} — {year}-{month:02d}-{day:02d} — lags={lags}")
                ax.legend(loc="upper right")
                ax.grid(True, linestyle=":", linewidth=0.5)

                plt.tight_layout()
                plt.show()

            #exit()
            

        errors = np.array(all_errors)

        print(f"Summary of squared prediction errors with window {window}:")
        #print(f"Count:    {errors.size}")
        print(f"Mean:     {errors.mean():.6e}")
        print(f"Median:   {np.median(errors):.6e}")
        #print(f"Std dev:  {errors.std():.6e}")
        #print(f"Min:      {errors.min():.6e}")
        #print(f"Max:      {errors.max():.6e}")
        #print("Percentiles:")
        #for q in (25, 50, 75):
        #    print(f"  {q:2d}th:   {np.percentile(errors, q):.6e}")
        print()

        res_means.append(errors.mean())
        res_medians.append(np.median(errors))


  
# assume you have two windows in params_volatility
n_windows = len(params_volatility)
# reshape the flat lists into shape (n_windows, len(H_values))
means_arr      = np.array(res_means).reshape(len(H_values), n_windows).transpose()
medians_arr    = np.array(res_medians).reshape(len(H_values), n_windows).transpose()

# map each row back to its window size
window_sizes = [p["window"] for p in params_volatility]

# ---- create the plot ----
fig, ax = plt.subplots(figsize=(10, 6))

for i, w in enumerate(window_sizes):
    ax.plot(H_values, means_arr[i],
           label=f'Mean error (window={w})',
           marker='s', linestyle='-',
           alpha=0.8)
    # ax.plot(H_values, medians_arr[i],
    #         label=f'Median error (window={w})',
    #         marker='o', linestyle='--',
    #         alpha=0.8)

# ---- styling ----
ax.set_xlabel('Hurst exponent $H$', fontsize=12)
ax.set_ylabel('Mean squared prediction error', fontsize=12)
ax.set_title(f'Prediction error', fontsize=14, pad=15)
ax.legend(title='Statistics and window', fontsize=10)
ax.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()

plt.show()
