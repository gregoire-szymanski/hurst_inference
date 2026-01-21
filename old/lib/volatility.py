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
    def __init__(self, delta, window, price_truncation="INFINITE"):
        # Ensure that the truncation method provided is one of the allowed types
        if price_truncation not in ["INFINITE", "STD3", "BIVAR3", "STD5", "BIVAR5"]:
            raise ValueError("Invalid truncation method. Choose one of: 'INFINITE', 'STD3', 'BIVAR3', 'STD5', 'BIVAR5'")

        # Store the parameters as instance variables
        self.delta = delta
        self.window = window
        self.price_truncation = price_truncation
        
    def compute(self, price):
        price = np.log(price)
        priceinc = price[1:] - price[:-1]

        truncationValue = np.inf
        if self.price_truncation == 'STD3':
            truncationValue = 3 * np.std(priceinc)
        elif self.price_truncation == 'STD5':
            truncationValue = 5 * np.std(priceinc)
        elif self.price_truncation == 'BIVAR3':
            bav = bipower_average_V(price, self.window, self.delta)
            truncationValue = 3 * np.sqrt(bav * self.delta)
        elif self.price_truncation == 'BIVAR5':
            bav = bipower_average_V(price, self.window, self.delta)
            truncationValue = 5 * np.sqrt(bav * self.delta)

        priceinc[np.abs(priceinc) > truncationValue] = 0

        # Realized variance cumulative sum
        rv = np.concatenate([[0], np.cumsum(priceinc**2)])

        # Average volatility (not necessarily needed for the returned Volatility)
        avgVol = rv[-1] / (self.delta * (len(price) - 1))
        
        # Windowed volatility estimate
        volatilities = (rv[self.window:] - rv[:-self.window]) / (self.delta * self.window)

        return Volatility(volatilities)




class VolatilityPattern:
    def __init__(self):
        self.current_pattern = None
        self.N_elements = 0

    def accumulate(self, vol):
        if isinstance(vol, list):
            # vol is a list of Volatility objects
            all_values = [v.get_values() for v in vol]
            # Normalize each by its mean
            all_values = [arr / np.mean(arr) for arr in all_values]
            # Sum across all given volatilities
            sum_values = np.sum(all_values, axis=0)

            # Add current pattern if it exists
            if self.current_pattern is not None:
                sum_values += self.current_pattern.get_values() 

            # Update current pattern
            self.current_pattern = Volatility(sum_values)
            self.N_elements += len(vol)
        else:
            if isinstance(vol, Volatility):
                # vol is a single Volatility object
                vol_values = vol.get_values()
            else:
                vol_values = vol
            vol_values = vol_values / np.mean(vol_values)

            # If we have a current pattern, add it
            if self.current_pattern is not None:
                sum_values = vol_values + self.current_pattern.get_values() 
            else:
                sum_values = vol_values

            self.current_pattern = Volatility(sum_values)
            self.N_elements += 1

    def get_pattern(self):
        if self.current_pattern is None or self.N_elements == 0:
            # No pattern accumulated yet
            return None
        # Return the averaged pattern
        return Volatility(self.current_pattern.get_values() / self.N_elements)
            



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
    all_values = [v / np.mean(v) for v in all_values]

    # Compute average across all vols
    avg_values = np.mean(all_values, axis=0)
    # Normalize so that average is 1
    avg_val_mean = np.mean(avg_values)
    if avg_val_mean != 0:
        avg_values = avg_values / avg_val_mean

    return Volatility(avg_values)
