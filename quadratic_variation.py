import numpy as np

def computeVolatilityIncrements(window, vol_truncation, volatilities, pattern):
    mean_volatilities = np.mean(volatilities)
    volatilities = volatilities / pattern / mean_volatilities

    volatilities_increments = volatilities[window:] - volatilities[:-window]

    if vol_truncation == 'STD3':
        truncationValue = 3 * np.std(volatilities_increments)
    elif vol_truncation == 'STD5':
        truncationValue = 5 * np.std(volatilities_increments)
    elif type(vol_truncation) == float:
        truncationValue = vol_truncation
    
    volatilities_increments[np.abs(volatilities_increments) > truncationValue] = 0
    
    return volatilities_increments


class QuadraticCovariationsEstimator:
    def __init__(self, window, N_lags = 10, vol_truncation="INFINITE"):
        self.vol_truncation = vol_truncation
        self.window = window
        self.N_lags = N_lags

        if (type(vol_truncation) != float) and (vol_truncation not in ["INFINITE", "STD3", "STD5"]):
            raise ValueError("Invalid truncation method. Choose one of: 'INFINITE', 'STD3', 'BIVAR3', 'STD5', 'BIVAR5'")    

    def precompute(self, volatilities, pattern):
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

        return volatilities_increments


    def conclude(self, volatilities_increments, first_lag_correction=True):
        covariations = np.zeros(self.N_lags)

        for lag in range(self.N_lags):
            if lag == 0:
                covariations[lag] = np.mean(volatilities_increments**2) 
            else:
                covariations[lag] = np.mean(volatilities_increments[(lag * self.window):] * volatilities_increments[: - (lag * self.window)])
        
        if first_lag_correction:
            covariations[0] = covariations[0][:len(covariations[1])]
            covariations[1] = covariations[0] + 2 * covariations[1]
            return covariations[1:]
        else:
            return covariations


    def DRV(self, volatilities_increments, first_lag_correction=True):
        covariations = []

        for lag in range(self.N_lags):
            if lag == 0:
                covariations.append(volatilities_increments**2)
            else:
                covariations.append(volatilities_increments[(lag * self.window):] * volatilities_increments[: - (lag * self.window)])
        
        if first_lag_correction:
            covariations[0] = covariations[0][:len(covariations[1])]
            covariations[1] = covariations[0] + 2 * covariations[1]
            return covariations[1:]
        else:
            return covariations

    def compute(self, volatilities, pattern):
        return self.conclude(self.precompute(volatilities, pattern))

    


class AsymptoticVarianceEstimator:
    def __init__(self, W_fun, Ln, Kn):
        """
        Parameters
        ----------
        W_fun : function
            A weighting function W_fun(L, Ln).
        Ln : int
            The maximum lag (positive and negative) to consider.
        Kn : int
            The window size for the moving average step.
        """
        self.W_fun = W_fun
        self.Ln = Ln
        self.Kn = Kn
        
    def correction(self, DRV):
        """
        Compute a moving average (correction) of DRV with a window of size Kn.
        
        Parameters
        ----------
        DRV : array-like
            Input data vector.
        
        Returns
        -------
        psi : np.ndarray
            The moving average of DRV with window size Kn. 
            Its size will be (len(DRV) - Kn + 1) if len(DRV) >= Kn.
        """
        DRV = np.asarray(DRV)
        if len(DRV) < self.Kn:
            raise ValueError("DRV length must be at least Kn.")
        # Simple moving average
        kernel = np.ones(self.Kn) / self.Kn
        psi = np.convolve(DRV, kernel, mode='valid')
        return psi

    def compute_term(self, psi, psi_prime, kn, kn_prime, L):
        """
        Compute a single term in the asymptotic variance estimator for a given lag L.
        
        Parameters
        ----------
        psi : np.ndarray
            A time series of corrected values.
        psi_prime : np.ndarray
            Another time series of corrected values.
        kn : float
            A scaling parameter.
        kn_prime : float
            Another scaling parameter.
        L : int
            The lag at which to compute the term.
        
        Returns
        -------
        float
            The computed term.
        """
        psi = np.asarray(psi)
        psi_prime = np.asarray(psi_prime)
        
        # Align psi and psi_prime based on lag L
        if L > 0:
            # psi_prime is shifted forward by L relative to psi
            # so we must drop the last L elements of psi and the first L elements of psi_prime
            psi_trunc = psi[:-L]
            psi_prime_trunc = psi_prime[L:]
        elif L < 0:
            # psi is shifted forward by -L relative to psi_prime
            # so we must drop the first -L elements of psi and the last -L elements of psi_prime
            shift = -L
            psi_trunc = psi[shift:]
            psi_prime_trunc = psi_prime[:-shift]
        else:
            # L = 0, no shift
            psi_trunc = psi
            psi_prime_trunc = psi_prime
        
        # Ensure equal lengths
        N = min(len(psi_trunc), len(psi_prime_trunc))
        if N == 0:
            return 0.0
        
        psi_trunc = psi_trunc[:N]
        psi_prime_trunc = psi_prime_trunc[:N]
        
        # Compute mean of product
        val = np.mean(psi_trunc * psi_prime_trunc)
        return val / np.sqrt(kn * kn_prime)

    def compute(self, psi, psi_prime, kn, kn_prime):
        """
        Compute the asymptotic variance estimate by summing over lags from -Ln to Ln.
        
        Parameters
        ----------
        psi : np.ndarray
            Corrected series (e.g., from self.correction).
        psi_prime : np.ndarray
            Another corrected series.
        kn : float
            A scaling parameter.
        kn_prime : float
            Another scaling parameter.
        
        Returns
        -------
        float
            The asymptotic variance estimate.
        """
        result = 0.0
        for L in range(-self.Ln, self.Ln + 1):
            w = self.W_fun(self.Ln,L)
            term = self.compute_term(psi, psi_prime, kn, kn_prime, L)
            result += w * term
        return result



    def compute0(self, psi, psi_prime, kn, kn_prime):
        w = self.W_fun(self.Ln,0)
        term = self.compute_term(psi, psi_prime, kn, kn_prime, 0)
        return w * term

    def compute_pos(self, psi, psi_prime, kn, kn_prime):
        result = 0.0
        for L in range(1, self.Ln + 1):
            w = self.W_fun(self.Ln,L)
            term = self.compute_term(psi, psi_prime, kn, kn_prime, L)
            result += w * term
        return result

