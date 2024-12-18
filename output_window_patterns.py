from step_params import *
from volatility import *
from timer import *
import matplotlib.pyplot as plt

# Ensure required modules for handling dates and data retrieval are imported
from datetime import datetime
from collections import defaultdict

print("Computing volatility patterns...")
timer = Timer(len(params_volatility), type="window")
timer.start()

# Create a single figure for all subplots
fig, axs = plt.subplots(len(params_volatility), figsize=(10, 5 * len(params_volatility)))

# Ensure axs is iterable even if there is only one parameter
if len(params_volatility) == 1:
    axs = [axs]

# Iterate over each volatility parameter
for i, param in enumerate(params_volatility):
    timer.step(i)

    # Initialize the overall pattern and yearly patterns dictionary
    pattern = VolatilityPattern()
    year_patterns = defaultdict(VolatilityPattern)

    # Iterate over each date (assumed to be tuples of year, month, day)
    for (year, month, day) in dates:
        try:
            # Retrieve volatility data for the given parameters
            vol = DH.get_data(FileTypeVolatility(asset, year, month, day, param["window"]))

            # Accumulate data into yearly and overall patterns
            year_patterns[year].accumulate(vol)
            pattern.accumulate(vol)
        except Exception as e:
            print(f"Error processing data for {year}-{month}-{day}: {e}")

    # Plot the overall pattern
    ax = axs[i]
    ax.plot(pattern.get_pattern().get_values(), label="Overall Pattern")

    # Plot the yearly patterns
    for year, year_pattern in year_patterns.items():
        ax.plot(year_pattern.get_pattern().get_values(), label=f"Year {year}")

    ax.set_title(f"Volatility Patterns - Window {param['window']}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility")
    ax.legend()

print(f"Volatility patterns computed in {timer.total_time():.2f}s.")

# Display all plots
plt.tight_layout()
plt.show()

