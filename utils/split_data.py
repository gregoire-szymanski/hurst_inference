import os
import re
from datetime import datetime, time

import numpy as np
import pandas as pd

input_folder = "/Users/gregoireszymanski/Documents/data/spy/price/1s/year_csv"
output_folder = "/Users/gregoireszymanski/Documents/data/spy/price/1s/c_daily_csv"  # Create folder if needed


def get_trading_days(year: int) -> pd.DatetimeIndex:
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("NYSE")
        schedule = cal.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
        return schedule.index.tz_localize(None)
    except Exception:
        from pandas.tseries.holiday import (
            AbstractHolidayCalendar,
            GoodFriday,
            Holiday,
            USLaborDay,
            USMartinLutherKingJr,
            USMemorialDay,
            USPresidentsDay,
            USThanksgivingDay,
        )
        from pandas.tseries.offsets import CustomBusinessDay

        class NYSEHolidayCalendar(AbstractHolidayCalendar):
            rules = [
                Holiday("NewYearsDay", month=1, day=1, observance=pd.tseries.holiday.nearest_workday),
                USMartinLutherKingJr,
                USPresidentsDay,
                GoodFriday,
                USMemorialDay,
                Holiday("Juneteenth", month=6, day=19, observance=pd.tseries.holiday.nearest_workday, start_date="2022-01-01"),
                Holiday("IndependenceDay", month=7, day=4, observance=pd.tseries.holiday.nearest_workday),
                USLaborDay,
                USThanksgivingDay,
                Holiday("Christmas", month=12, day=25, observance=pd.tseries.holiday.nearest_workday),
            ]

        cal = NYSEHolidayCalendar()
        holidays = cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
        bday = CustomBusinessDay(holidays=holidays)
        return pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq=bday)


os.makedirs(output_folder, exist_ok=True)

list_year_files = [
    fn
    for fn in sorted(os.listdir(input_folder))
    if re.match(r"^prices_(\d{4})\.csv$", fn)
]

for year_file in list_year_files:
    m = re.match(r"^prices_(\d{4})\.csv$", year_file)
    if not m:
        continue
    year = int(m.group(1))
    trading_days = get_trading_days(year)

    year_path = os.path.join(input_folder, year_file)
    year_df = pd.read_csv(year_path, dtype=np.float64)
    values = year_df.to_numpy(copy=False)

    if values.shape[1] != len(trading_days):
        raise ValueError(
            f"Calendar mismatch for {year}: data columns={values.shape[1]}, "
            f"trading days={len(trading_days)}"
        )

    for col_idx, day in enumerate(trading_days):
        prices = values[:, col_idx]
        dt = pd.date_range(
            start=datetime.combine(day.date(), time(9, 30)),
            periods=prices.shape[0],
            freq="S",
        )
        out_df = pd.DataFrame({"DT": dt, "Price": prices})
        out_name = f"spy_{day.strftime('%Y_%m_%d')}.csv"
        out_path = os.path.join(output_folder, out_name)
        out_df.to_csv(out_path, index=False)
