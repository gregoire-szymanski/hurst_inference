print("Initialisation...")

# Identification
identificator = "opt_5s_120w_150w"

if identificator ==  "opt_5s_120w_150w":
    # Global parameters
    asset = 'spy'
    subsampling = 5
    delta = 1.0 / (252.0 * 23400) * subsampling  # Time increment

    # Volatility and quadratic variation estimation parameters
    price_truncation_method = 'BIVAR3'
    vol_truncation_method = 'STD3'

    params_volatility = [
        {'window': 120, 'N_lags': 6},
        {'window': 150, 'N_lags': 4}
    ]

    # Asymptotic variance estimation
    Ln = 180
    Kn = 720
    W_fun_id = 'parzen'

else:
    raise ValueError("Unknown identificator")

