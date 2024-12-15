import numpy as np
import pandas as pd

def bipower_average_V(price, window, delta):
    n = len(price)
    if n <= 2 * window:  # Ensure there's enough data
        print("Not enough data points.")
        return -1.0

    # Compute price increments over the given window
    price_increments = price[window:] - price[:-window]

    # Calculate bipower average volatility
    sum_ = np.sum(np.abs(price_increments[window:] * price_increments[:-window]))

    # Calculate the final result
    mean = sum_ / (n - 2 * window)
    return (mean / (delta * window)) * (np.pi / 2)


class Volatility:
    def __init__(self, values):
        """
        values: array-like of computed volatility values
        """
        self.values = np.array(values)
        
    def get_values(self):
        return self.values
        
    def rv(self, delta):
        """
        Realized variance as sum of values * delta.
        """
        return np.sum(self.values) * delta


class VolatilityEstimator:
    def __init__(self, delta, window, subsampling, price_truncation="INFINITE"):
        # Ensure that the truncation method provided is one of the allowed types
        if price_truncation not in ["INFINITE", "STD3", "BIVAR3", "STD5", "BIVAR5"]:
            raise ValueError("Invalid truncation method. Choose one of: 'INFINITE', 'STD3', 'BIVAR3', 'STD5', 'BIVAR5'")

        # Store the parameters as instance variables
        self.delta = delta
        self.window = window
        self.subsampling = subsampling
        self.price_truncation = price_truncation
        
    def compute(self, price):
        price = price[::self.subsampling]
        price = np.log(price)
        priceinc = price[1:] - price[:-1]

        truncationValue = np.inf
        if self.price_truncation == 'STD3':
            truncationValue = 3 * np.std(priceinc)
        elif self.price_truncation == 'STD5':
            truncationValue = 5 * np.std(priceinc)
        elif self.price_truncation == 'BIVAR3':
            bav = bipower_average_V(price, self.window, self.delta * self.subsampling)
            truncationValue = 3 * np.sqrt(bav * self.delta * self.subsampling)
        elif self.price_truncation == 'BIVAR5':
            bav = bipower_average_V(price, self.window, self.delta * self.subsampling)
            truncationValue = 5 * np.sqrt(bav * self.delta * self.subsampling)

        priceinc[np.abs(priceinc) > truncationValue] = 0

        # Realized variance cumulative sum
        rv = np.concatenate([[0], np.cumsum(priceinc**2)])

        # Average volatility (not necessarily needed for the returned Volatility)
        avgVol = rv[-1] / (self.delta * self.subsampling * (len(price) - 1))
        
        # Windowed volatility estimate
        volatilities = (rv[self.window:] - rv[:-self.window]) / (self.delta * self.subsampling * self.window)

        return Volatility(volatilities)


def volatility_pattern(vols):
    """
    vols: a list of Volatility objects with the same DT.
    
    Create a new Volatility that averages all the vols.values and then
    normalizes them so that their average value is 1.
    """
    if not vols:
        raise ValueError("No Volatility objects provided.")

    # Extract values from each
    all_values = [v.get_values() for v in vols]
    # Compute average across all vols
    avg_values = np.mean(all_values, axis=0)
    # Normalize so that average is 1
    avg_val_mean = np.mean(avg_values)
    if avg_val_mean != 0:
        avg_values = avg_values / avg_val_mean

    return Volatility(avg_values)
