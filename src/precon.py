"""Read PRECON household files, mark gap-filled stretches, and build an hourly table.

PRECON (Nadeem and Arshad, e-Energy 2019) gives 1-minute kW readings for 42
houses in Lahore. The meters switch off during power outages, yet the
published files have no missing minutes, so gaps were filled in some way that
the paper does not describe. Stretches where every column moves on an exact
straight line (or stays constant) for 10 minutes or more are treated here as
filled, not measured, and are left out of every fit.

usage: python precon.py <folder with HouseN.csv files>
writes data/derived/hourly.csv.gz and data/derived/house_summary.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "data", "derived")
MIN_FILL = 10          # minutes; shorter straight runs are left as measured
MIN_VALID = 0.8        # an hour needs 80 % measured minutes to be kept


def ac_columns(cols):
    return [c for c in cols if c.upper().startswith("AC")]


def filled_mask(X, min_len=MIN_FILL):
    """True for minutes inside a run where all columns lie on one straight line."""
    lin = np.all(np.abs(np.diff(X, 2, axis=0)) < 1e-6, axis=1)
    # second difference at i covers rows i, i+1, i+2
    run = np.r_[False, lin, False]
    mask = np.zeros(len(X), bool)
    i, n = 0, len(run)
    while i < n:
        if run[i]:
            j = i
            while j < n and run[j]:
                j += 1
            # run[i:j] are second differences i-1 .. j-2 -> rows i-1 .. j
            if (j - i) + 2 >= min_len:
                mask[max(i - 1, 0):min(j + 1, len(X))] = True
            i = j
        else:
            i += 1
    return mask


def load_house(path):
    d = pd.read_csv(path)
    t = pd.to_datetime(d["Date_Time"])
    cols = [c for c in d.columns if c != "Date_Time"]
    X = d[cols].to_numpy(float)
    fill = filled_mask(X)
    acs = ac_columns(cols)
    out = pd.DataFrame({"time": t, "usage_kw": d["Usage_kW"].to_numpy(float)})
    out["ac_kw"] = d[acs].sum(axis=1).to_numpy(float) if acs else np.nan
    out["ups_kw"] = d["UPS_kW"].to_numpy(float) if "UPS_kW" in cols else np.nan
    out["filled"] = fill
    return out, acs


def to_hourly(m):
    m = m.copy()
    for c in ("usage_kw", "ac_kw", "ups_kw"):
        m.loc[m["filled"], c] = np.nan
    m["hour"] = m["time"].dt.floor("h")
    g = m.groupby("hour")
    h = g[["usage_kw", "ac_kw", "ups_kw"]].mean()
    h["valid_frac"] = 1 - g["filled"].mean()
    h.loc[h["valid_frac"] < MIN_VALID, ["usage_kw", "ac_kw", "ups_kw"]] = np.nan
    return h.reset_index().rename(columns={"hour": "time"})


def main(folder):
    os.makedirs(DERIVED, exist_ok=True)
    files = sorted(glob.glob(os.path.join(folder, "House*.csv")),
                   key=lambda s: int(os.path.basename(s)[5:-4]))
    rows, summ = [], []
    for f in files:
        hid = int(os.path.basename(f)[5:-4])
        m, acs = load_house(f)
        h = to_hourly(m)
        h.insert(0, "house", hid)
        rows.append(h)
        meas = ~m["filled"]
        summ.append(dict(
            house=hid, start=m["time"].min(), end=m["time"].max(),
            minutes=len(m), filled_minutes=int(m["filled"].sum()),
            n_ac_cols=len(acs), has_ups=bool(m["ups_kw"].notna().any()),
            mean_kw=round(m.loc[meas, "usage_kw"].mean(), 3),
            ac_share=(round(m.loc[meas, "ac_kw"].sum() / m.loc[meas, "usage_kw"].sum(), 3)
                      if acs else np.nan),
            ac_gt_usage_frac=(round(float((m.loc[meas, "ac_kw"] > m.loc[meas, "usage_kw"] + 0.05).mean()), 4)
                              if acs else np.nan)))
        print(summ[-1])
    pd.concat(rows).to_csv(os.path.join(DERIVED, "hourly.csv.gz"), index=False, float_format="%.4f")
    pd.DataFrame(summ).to_csv(os.path.join(DERIVED, "house_summary.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
