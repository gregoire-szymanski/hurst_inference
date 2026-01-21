
import pandas as pd

FILE_PATH = "/Users/gregoireszymanski/Documents/data/vol/oxfordmanrealizedvolatilityindices.csv"
data = pd.read_csv(FILE_PATH, memory_map=True)

symbols = data["Symbol"].unique()
print(symbols)

symbol = ".SPX"
filtered = data[data["Symbol"] == symbol][["Date", "rv"]]

print(filtered)