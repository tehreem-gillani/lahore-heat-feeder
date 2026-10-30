"""Virtual feeder: PRECON houses summed minute by minute.

Of the 41 houses received (House 35 was not in the download), only houses
with at least half of every month measured are used, so that no month of a
house is invented from other months.

The houses are spread across Lahore and are not on one transformer in
reality; putting them on one is the study's main assumption.

Minutes that were gap-filled in the published files (see precon.py) are
replaced by the same house's mean for that month and clock hour, taken from
its measured minutes only. This treats every house as supplied, i.e. it
models the feeder as if there were no load shedding. The share of imputed
energy is kept for every day so that study days can be chosen with little
imputation.

usage: python feeder.py <folder with HouseN.csv files>
writes data/derived/feeder_1min.csv.gz and data/derived/feeder_days.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import precon

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "data", "derived")
START, END = "2018-06-01 00:00", "2019-05-31 23:59"
MIN_MONTH_COVERAGE = 0.5   # a house joins the feeder only if every month is at least half measured


def house_series(path, index):
    m, _ = precon.load_house(path)
    s = pd.Series(m["usage_kw"].to_numpy(), index=m["time"])
    s = s[~s.index.duplicated()].reindex(index)
    filled = pd.Series(m["filled"].to_numpy(), index=m["time"])
    filled = filled[~filled.index.duplicated()].reindex(index)
    # minutes outside the house's recording period count as missing too
    missing = filled.fillna(True).astype(bool).to_numpy()
    meas = s.where(~missing)
    key_m, key_h = index.month, index.hour
    clim = meas.groupby([key_m, key_h]).mean()
    fill_vals = clim.reindex(list(zip(key_m, key_h))).to_numpy()
    out = np.where(missing, fill_vals, s.to_numpy())
    return out, missing


def main(folder):
    os.makedirs(DERIVED, exist_ok=True)
    index = pd.date_range(START, END, freq="min")
    files = sorted(glob.glob(os.path.join(folder, "House*.csv")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    total = np.zeros(len(index))
    imputed = np.zeros(len(index))
    months = index.to_period("M")
    rows = []
    for f in files:
        hid = int(os.path.basename(f)[5:-4])
        v, miss = house_series(f, index)
        cov = 1 - pd.Series(miss, index=index).groupby(months).mean()
        keep = cov.min() >= MIN_MONTH_COVERAGE
        rows.append(dict(house=hid, min_month_coverage=round(cov.min(), 3),
                         worst_month=str(cov.idxmin()), in_feeder=keep,
                         imputed_minutes=int(miss.sum())))
        if not keep:
            continue
        assert not np.isnan(v).any()
        total += v
        imputed += np.where(miss, v, 0.0)
    hs = pd.DataFrame(rows)
    hs.to_csv(os.path.join(DERIVED, "feeder_houses.csv"), index=False)
    print("houses in feeder", int(hs.in_feeder.sum()), "of", len(hs),
          "; left out:", hs.loc[~hs.in_feeder, "house"].tolist())
    df = pd.DataFrame({"time": index, "load_kw": total, "imputed_kw": imputed})
    df.to_csv(os.path.join(DERIVED, "feeder_1min.csv.gz"), index=False, float_format="%.4f")
    day = df.groupby(df["time"].dt.date).agg(
        energy_kwh=("load_kw", lambda x: x.sum() / 60),
        imputed_kwh=("imputed_kw", lambda x: x.sum() / 60),
        peak_kw=("load_kw", "max"))
    day["imputed_share"] = day["imputed_kwh"] / day["energy_kwh"]
    day.index.name = "date"
    day.to_csv(os.path.join(DERIVED, "feeder_days.csv"), float_format="%.4f")


if __name__ == "__main__":
    main(sys.argv[1])
