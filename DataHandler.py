import os
import re
import datetime
import pandas as pd

class FileType:
    def __init__(self, asset='xxx', year=None, month=None, day=None, data_type=None, norm=None, param=None, window=None):
        self.asset = asset
        self.year = year
        self.month = month
        self.day = day
        # data_type must be one of 'vol', 'norm_vol', 'qv'
        if data_type not in ['vol', 'norm_vol', 'qv']:
            raise ValueError("data_type must be one of 'vol', 'norm_vol', 'qv'")
        self.data_type = data_type
        self.norm = norm
        self.param = param
        self.window = window

    def to_string(self):
        # returns datatype_norm_param_window_
        # The instructions are not completely clear, but we can form a string like:
        # data_type, followed by norm if present, param if present, window if present
        # Example: "vol_norm_param5_window10"
        # If not present, skip them.
        
        parts = [self.data_type]
        if self.norm is not None:
            parts.append(self.norm)
        if self.param is not None:
            parts.append(self.param)
        if self.window is not None:
            parts.append(str(self.window))
        return "_".join(parts)


class DataHandler:
    def __init__(self, prices_folder, tmp_folder):
        # Check that prices_folder exist
        if not os.path.isdir(prices_folder):
            raise FileNotFoundError(f"Prices folder '{prices_folder}' does not exist.")
        # Check that tmp_folder exist (if not, maybe create it)
        if not os.path.isdir(tmp_folder):
            os.makedirs(tmp_folder)

        self.prices_folder = prices_folder
        self.tmp_folder = tmp_folder

        # Make list of all files in prices_folder
        self.price_files = os.listdir(prices_folder)

        # Check that all price files follow pattern xxx_YYYY-MM-DD.csv
        pattern = re.compile(r'^[A-Za-z0-9]{3}_\d{4}-\d{2}-\d{2}\.csv$')
        for f in self.price_files:
            if not pattern.match(f):
                raise ValueError(f"Price file '{f}' does not match the pattern 'xxx_YYYY-MM-DD.csv'")
        
        self.tmp_files_created = []
    
    def remove_date(self, full_date): # Format of full_date: YYYY-MM-DD
        # Rebuild self.price_files list, excluding any file that contains the given full_date
        self.price_files = [f for f in self.price_files if full_date not in f]

    def __del__(self):
        # destructor of the class where all tmp files created are removed
        for f in self.tmp_files_created:
            if os.path.exists(f):
                os.remove(f)

    def get_price(self, asset, year, month, day):
        # read csv associated with this date
        # The filename should be something like xxx_YYYY-MM-DD.csv
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        filename = f"{asset}_{date_str}.csv"
        fullpath = os.path.join(self.prices_folder, filename)
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"No price file found for {asset} on {date_str}")
        df = pd.read_csv(fullpath)
        # Ensure DT is datetime type
        if 'DT' in df.columns:
            df['DT'] = pd.to_datetime(df['DT'])
        else:
            raise ValueError(f"CSV {filename} does not have 'DT' column.")

        if 'price' not in df.columns:
            raise ValueError(f"CSV {filename} does not have 'price' column.")

        return df

    def index_to_date(self, asset, index):
        # returns (year, month, day)
        # Without a defined mapping, we must guess.
        # Let's assume index corresponds to the nth file sorted by date for that asset.
        # We'll sort the files for that asset by date and pick the index-th.
        asset_files = [f for f in self.price_files if f.startswith(asset+"_")]
        # sort by date
        asset_files_sorted = sorted(asset_files, key=lambda x: x.split('_')[1].replace('.csv', ''))
        if index < 0 or index >= len(asset_files_sorted):
            raise IndexError("Index out of range for asset.")
        f = asset_files_sorted[index]
        # format: xxx_YYYY-MM-DD.csv
        date_part = f.split('_')[1].replace('.csv', '')
        y, m, d = date_part.split('-')
        return int(y), int(m), int(d)

    def date_to_index(self, asset, year, month, day):
        # the inverse of index_to_date
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        asset_files = [f for f in self.price_files if f.startswith(asset+"_")]
        # sort by date
        asset_files_sorted = sorted(asset_files, key=lambda x: x.split('_')[1].replace('.csv', ''))
        for i, f in enumerate(asset_files_sorted):
            date_part = f.split('_')[1].replace('.csv', '')
            if date_part == date_str:
                return i
        raise ValueError("Specified date not found for the given asset.")

    def following_date(self, asset, year, month, day):
        # return (year, month, day) + 1 for asset
        # We'll just increment by one day using datetime and see if that date exists
        current_date = datetime.date(year, month, day)
        next_date = current_date + datetime.timedelta(days=1)
        # We should verify that this next_date exists for the asset
        # If not, maybe just return the next_date anyway. 
        # The instructions aren't clear, but let's return the next available file date if it exists.
        # If it does not exist, raise an error or return None.
        
        next_date_str = next_date.strftime("%Y-%m-%d")
        filename = f"{asset}_{next_date_str}.csv"
        if os.path.exists(os.path.join(self.prices_folder, filename)):
            return next_date.year, next_date.month, next_date.day
        else:
            # If we must strictly follow next available date in files, we can loop forward until we find one.
            # For simplicity, just return the next date, even if file not present. 
            # Adjust as needed based on actual requirements.
            return next_date.year, next_date.month, next_date.day

    def save_data(self, filetype_obj, data):
        # Save into tmp folder
        # Use filetype_obj.to_string() for filename
        # We'll prepend something like asset_year-month-day_ and then the filetype string.
        date_str = f"{filetype_obj.year:04d}-{filetype_obj.month:02d}-{filetype_obj.day:02d}"
        filename = f"{filetype_obj.asset}_{date_str}_{filetype_obj.to_string()}.csv"
        filepath = os.path.join(self.tmp_folder, filename)
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            # If data is not a DataFrame, handle accordingly. 
            # For simplicity, assume it's something we can write with CSV.
            # Let's assume data is a list of dict or tuples.
            # We'll just write a generic CSV:
            import csv
            if len(data) > 0 and isinstance(data[0], dict):
                fieldnames = data[0].keys()
            else:
                fieldnames = ['value']
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    if isinstance(row, dict):
                        writer.writerow(row)
                    else:
                        writer.writerow({'value': row})

        self.tmp_files_created.append(filepath)

    def get_data(self, filetype_obj):
        # Get from tmp folder
        date_str = f"{filetype_obj.year:04d}-{filetype_obj.month:02d}-{filetype_obj.day:02d}"
        filename = f"{filetype_obj.asset}_{date_str}_{filetype_obj.to_string()}.csv"
        filepath = os.path.join(self.tmp_folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No such tmp data file.")
        # Return as DataFrame
        return pd.read_csv(filepath)
