import os
import re
import datetime
import pandas as pd
import csv
import numpy as np

class FileType:
    def __init__(self, subfolder='', asset='xxx', year=None, month=None, day=None):
        self.subfolder = subfolder
        self.asset = asset
        self.year = year
        self.month = month
        self.day = day
        
        if len(self.subfolder) and self.subfolder[-1] != '/':
            self.subfolder += '/'

    def to_string(self):
        if self.day is None:
            return self.subfolder + self.asset
        else:
            return self.subfolder + self.asset + "_" + f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


class DataHandler:
    def __init__(self, prices_folder, tmp_folder):
        prices_folder = os.path.expanduser(prices_folder)
        tmp_folder = os.path.expanduser(tmp_folder)

        # Check that prices_folder exists
        if not os.path.isdir(prices_folder):
            raise FileNotFoundError(f"Prices folder '{prices_folder}' does not exist.")
        
        # Check that tmp_folder exists (if not, create it)
        if not os.path.isdir(tmp_folder):
            os.makedirs(tmp_folder)

        self.prices_folder = prices_folder
        self.tmp_folder = tmp_folder

        # Make list of all files in prices_folder
        self.price_files = os.listdir(prices_folder)
        self.price_files.sort()

        # Check that all price files follow pattern xxx_YYYY-MM-DD.csv
        pattern = re.compile(r'^[A-Za-z0-9]{3}_\d{4}-\d{2}-\d{2}\.csv$')
        for f in self.price_files:
            if not pattern.match(f):
                raise ValueError(f"Price file '{f}' does not match the pattern 'xxx_YYYY-MM-DD.csv'")
        
        self.tmp_files_created = []
    
    def dates(self, asset=None):
        if asset is None:
            return np.unique([f.split('_')[1].replace('.csv', '') for f in self.price_files]).to_list()
        else:
            # Get all dates
            all_price_files = [f for f in self.price_files if f.startswith(asset+'_')]
            return [f.split('_')[1].replace('.csv','') for f in all_price_files]

    def remove_date(self, full_date): 
        # Format of full_date: YYYY-MM-DD
        self.price_files = [f for f in self.price_files if full_date not in f]

    def __del__(self):
        # destructor of the class where all tmp files created are removed
        for f in self.tmp_files_created:
            if not f["save"] and os.path.exists(f["path"]):
                os.remove(f["path"])

    def get_price(self, asset, year=None, month=None, day=None):
        # read csv associated with this date
        # The filename should be something like xxx_YYYY-MM-DD.csv
        
        filename = asset
        if day is not None:
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            filename = f"{asset}_{date_str}.csv"
        fullpath = os.path.join(self.prices_folder, filename)
        
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"No price file found for {asset} on {date_str}")
        
        return pd.read_csv(fullpath)

    def save_data(self, filetype_obj, data, save=True):
        # Use filetype_obj.to_string() for filename

        filepath = os.path.join(self.tmp_folder, filetype_obj.to_string())
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath + ".csv", index=False)
            self.tmp_files_created.append({"save":save, "path":filepath+ ".csv" })
        elif isinstance(data, np.ndarray):
            np.save(filepath + ".npy", data)
            self.tmp_files_created.append({"save": save, "path": filepath + ".npy"})
        elif isinstance(data, list):
            np.save(filepath + ".npy", np.array(data))
            self.tmp_files_created.append({"save": save, "path": filepath + ".npy"})
        else:
            raise TypeError("Data must be a pandas DataFrame or a numpy array")



    def get_data(self, filetype_obj):
        # Retrieve the data file
        filename = os.path.join(self.tmp_folder, filetype_obj.to_string())
        if os.path.exists(filename + ".csv"):
            return pd.read_csv(filename + ".csv")
        elif os.path.exists(filename + ".npy"):
            return np.load(filename + ".npy")
        else:
            raise FileNotFoundError(f"No such tmp data file: {filename}")
        



