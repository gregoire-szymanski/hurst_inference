import os
import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal # pip install pandas-market-calendars

# Input folder path and output base folder
input_folder = "/Users/gregoire.szymanski/Documents/data/year"
output_base_folder = "/Users/gregoire.szymanski/Documents/data/day"

# List of years
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

# Fetch NYSE trading calendar
nyse = mcal.get_calendar('NYSE')

# Iterate through each year
for year in years:
    # Create subfolder for the year
    year_folder = os.path.join(output_base_folder, str(year))
    year_folder = output_base_folder
    os.makedirs(year_folder, exist_ok=True)

    # Input file path for the current year
    input_file = os.path.join(input_folder, f"prices_{year}.csv")

    # Load the CSV file
    df = pd.read_csv(input_file)

    # Generate a list of NYSE trading days for the year
    schedule = nyse.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
    trading_days = schedule.index.to_pydatetime()

    # Create a mapping from trading day index to actual date
    trading_day_map = {i + 1: trading_day for i, trading_day in enumerate(trading_days)}


    # Iterate through each trading day (column in the CSV)
    for trading_day in df.columns:
        # Extract prices for the trading day
        prices = df[trading_day]

        # Parse trading day index
        trading_day_index = int(trading_day[1:])

        # Get the corresponding date from the trading day map
        start_date = trading_day_map[trading_day_index]

        # Generate timestamps for the trading day
        timestamps = [
            start_date + timedelta(seconds=i) + timedelta(seconds=34200)
            for i in range(len(prices))
        ]

        # Format timestamps into the "YYYY-MM-DD HH:MM:SS" format
        formatted_timestamps = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]

        # Create a DataFrame with the formatted timestamps and prices
        day_df = pd.DataFrame({"DT": formatted_timestamps, "Price": prices})

        # Construct the output file name
        output_file = os.path.join(year_folder, f"spy_{start_date.strftime('%Y-%m-%d')}.csv")

        # Save the DataFrame to a CSV file
        day_df.to_csv(output_file, index=False)
    print(output_file.split('/')[-1])

print("Splitting complete. Files are saved in respective yearly folders.")
