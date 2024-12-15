import numpy as np

class QuadraticCovariationsEstimator:
    def __init__(self, window, N_lags = 10, vol_truncation="INFINITE"):
        self.vol_truncation = vol_truncation
        self.window = window
        self.N_lags = N_lags

        if (type(vol_truncation) != float) and (vol_truncation not in ["INFINITE", "STD3", "STD5"]):
            raise ValueError("Invalid truncation method. Choose one of: 'INFINITE', 'STD3', 'BIVAR3', 'STD5', 'BIVAR5'")
        
    def compute(self, volatilities, pattern):
        mean_volatilities = np.mean(volatilities)
        volatilities = volatilities / pattern / mean_volatilities

        volatilities_increments = volatilities[self.window:] - volatilities[:-self.window]

        if self.vol_truncation == 'STD3':
            truncationValue = 3 * np.std(volatilities_increments)
        elif self.vol_truncation == 'STD5':
            truncationValue = 5 * np.std(volatilities_increments)
        elif type(self.vol_truncation) == float:
            truncationValue = self.vol_truncation
        
        volatilities_increments[np.abs(volatilities_increments) > truncationValue] = 0

        covariations = np.zeros(self.N_lags)

        for lag in range(self.N_lags):
            if lag == 0:
                covariations[lag] = np.mean(volatilities_increments**2) 
            else:
                covariations[lag] = np.mean(volatilities_increments[(lag * self.window):] * volatilities_increments[: - (lag * self.window)])
        
        return covariations
    



