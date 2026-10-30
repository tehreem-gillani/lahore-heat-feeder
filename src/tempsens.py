"""How much more electricity a Lahore house uses per degree of heat (measured).

For every house, daily mean load is fitted against daily mean air temperature
with a change-point model

    load = base + slope * max(0, T - T_b)

where T_b (the balance temperature) is searched on a 0.5 K grid. The fit is
made on even ISO weeks and tested on odd weeks, so the test days are unseen
but cover the same seasons. The no-temperature baseline for the test is the
house's mean load on its training days.

Only measured hours are used (gap-filled minutes are removed in precon.py)
and a day needs at least 20 valid hours.
"""
import os

import numpy as np
import pandas as pd

import weather

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "data", "derived")
TB_GRID = np.arange(15.0, 32.01, 0.5)
MIN_HOURS = 20
MIN_DAYS = 100


def fit_cp(T, y):
    best = None
    for tb in TB_GRID:
        X = np.c_[np.ones_like(T), np.maximum(0.0, T - tb)]
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = float(((y - X @ b) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, tb, b)
    _, tb, b = best
    return tb, float(b[0]), float(b[1])


def predict(T, tb, base, slope):
    return base + slope * np.maximum(0.0, T - tb)


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def daily_table():
    h = pd.read_csv(os.path.join(DERIVED, "hourly.csv.gz"), parse_dates=["time"])
    w = weather.load()
    d = h.merge(w, on="time", how="inner")
    d["date"] = d["time"].dt.normalize()
    day = d.groupby(["house", "date"]).agg(
        usage=("usage_kw", "mean"), ac=("ac_kw", "mean"),
        n=("usage_kw", "count"), T=("t_air", "mean")).reset_index()
    day = day[day["n"] >= MIN_HOURS].copy()
    day["train"] = day["date"].dt.isocalendar().week.astype(int) % 2 == 0
    return day


def fit_all(day, col):
    rows = []
    for hid, g in day.groupby("house"):
        g = g.dropna(subset=[col])
        tr, te = g[g.train], g[~g.train]
        if len(tr) < MIN_DAYS // 2 or len(te) < MIN_DAYS // 2:
            continue
        tb, base, slope = fit_cp(tr["T"].to_numpy(), tr[col].to_numpy())
        yhat = predict(te["T"].to_numpy(), tb, base, slope)
        y = te[col].to_numpy()
        rows.append(dict(
            house=hid, days_train=len(tr), days_test=len(te), t_balance=tb,
            base_kw=round(base, 3), slope_w_per_k=round(slope * 1000, 1),
            r2_test=round(r2(y, yhat), 3),
            mae_test_kw=round(float(np.abs(y - yhat).mean()), 3),
            mae_baseline_kw=round(float(np.abs(y - tr[col].mean()).mean()), 3)))
    return pd.DataFrame(rows)


def main():
    day = daily_table()
    tot = fit_all(day, "usage")
    ac = fit_all(day, "ac")
    tot.to_csv(os.path.join(DERIVED, "temp_sensitivity_total.csv"), index=False)
    ac.to_csv(os.path.join(DERIVED, "temp_sensitivity_ac_circuits.csv"), index=False)
    for name, t in (("total load", tot), ("metered AC circuits", ac)):
        print(f"{name}: houses {len(t)}, slope median {t.slope_w_per_k.median():.0f} W/K "
              f"(IQR {t.slope_w_per_k.quantile(.25):.0f}-{t.slope_w_per_k.quantile(.75):.0f}), "
              f"T_b median {t.t_balance.median():.1f} C, test R2 median {t.r2_test.median():.2f}, "
              f"MAE {t.mae_test_kw.median():.3f} vs baseline {t.mae_baseline_kw.median():.3f} kW")
    return tot, ac


if __name__ == "__main__":
    main()
