"""NASA POWER hourly file for Lahore (31.52 N, 74.36 E), 1 Jun 2018 to 31 May 2019.

POWER gives hours in local solar time (LST), with LST = UTC + round(lon/15) h.
For Lahore round(74.36/15) = 5, and Pakistan Standard Time is UTC+5 with no
daylight saving in 2018-19, so LST hours are already clock hours.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "..", "data", "raw", "nasa_power_lahore_2018_2019_LST.csv")
LON = 74.36
UTC_OFFSET = 5


def load(path=FILE):
    with open(path) as f:
        lines = f.read().splitlines()
    start = lines.index("-END HEADER-") + 1
    d = pd.read_csv(path, skiprows=start)
    if (d[["T2M", "ALLSKY_SFC_SW_DWN"]] <= -999).any().any():
        raise ValueError("missing values (-999) in POWER file")
    shift = UTC_OFFSET - round(LON / 15)
    t = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY, hour=d.HR)) + pd.Timedelta(hours=shift)
    return pd.DataFrame({"time": t, "t_air": d["T2M"].to_numpy(float),
                         "ghi": d["ALLSKY_SFC_SW_DWN"].to_numpy(float)})
