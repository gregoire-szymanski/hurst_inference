import pandas as pd

class Price:
    def __init__(self, df):
        # Ensure that DT is datetime
        if 'DT' not in df.columns:
            raise ValueError("DataFrame must have a 'DT' column.")

        df['DT'] = pd.to_datetime(df['DT'], errors='coerce')
        if df['DT'].isnull().any():
            raise ValueError("Some DT values could not be converted to datetime.")

        if 'price' not in df.columns:
            raise ValueError("DataFrame must have a 'price' column.")

        self.df = df.sort_values('DT').reset_index(drop=True)  # Ensure sorted by DT

    def trading_halt(self, duration=300):
        """
        Check for trading halts. A trading halt is identified if the gap between 
        consecutive timestamps is greater than 'duration' seconds.

        Returns:
            A Pandas Series of time gaps (in seconds) that exceed the duration.
            If empty, no halts are present.
        """
        time_diffs = self.df['DT'].diff().dt.total_seconds().fillna(0)
        halts = time_diffs[time_diffs > duration]
        return halts

    def subsample(self, sub=5):
        """
        Subsample the data every 'sub' seconds. The method sets 'DT' as index,
        resamples the data at 'sub' second intervals, taking the last 
        observed price in each interval.
        """
        self.df = (self.df.set_index('DT')
                         .resample(f'{sub}S')
                         .last()
                         .dropna()
                         .reset_index())

    def get_price(self):
        """
        Returns:
            Numpy array of the prices.
        """
        return self.df['price'].values

    def get_increments(self):
        """
        Returns:
            Numpy array of the increments (differences) of consecutive prices.
        """
        increments = self.df['price'].diff().dropna().values
        return increments
