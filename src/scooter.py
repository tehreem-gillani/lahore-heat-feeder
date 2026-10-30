"""Home charging of electric scooters: energy per day and charging rules.

Energy per km at the battery comes from Project 2 (e2w-controller-wearout):
the same 102 kg, 72 V scooter driving WMTC part 1 with UN GTR No. 2 road
load and an assumed 0.88 motor + controller efficiency gives 21.3 Wh/km
(src/vehicle.py there; see energy_from_project2() below). As a rough
cross-check, the Eveon Pronto sold in Pakistan (72 V, 32 Ah, 90 km range,
PakWheels, Jan 2025) works out at about 26 Wh/km.

Assumptions (stated in the report and varied):
  charger efficiency 0.85; charger power 0.5 kW from the grid
  (the Pronto's 4 A charger is about 0.35 kW);
  commuter: Monday to Saturday, 24 km/day (two 12 km rides, 08:00 and
           17:30 as in Project 2), plugs in on arrival, 18:00-19:30
           (uniform), must be full by 07:45;
  delivery: every day, 96 km/day (twelve 8 km rides, 11:00 to 22:20),
           plugs in 22:30-23:30 (uniform), must be full by 10:45.
Every scooter needs that day's energy back before it leaves the next day.
"""
import os
import sys

import numpy as np
import pandas as pd

WH_PER_KM_BATT = 21.3
ETA_CHARGER = 0.85
P_CHARGER = 0.5            # kW at the grid
PROFILES = {
    # km/day, riding days (0 = Monday), plug-in window (h), departure next day (h)
    "commuter": dict(km=24.0, days=(0, 1, 2, 3, 4, 5), plug=(18.0, 19.5), leave=7.75),
    "delivery": dict(km=96.0, days=(0, 1, 2, 3, 4, 5, 6), plug=(22.5, 23.5), leave=10.75),
}
# Time-of-use peak window by month (end hour), NEPRA LESCO tariff determination
# TRF-603 & TRF-604, 7 Jan 2026, schedule of tariff, definitions cl. 10:
# Dec-Feb 17-21, Mar-May 18-22, Jun-Aug 19-23, Sep-Nov 18-22.
PEAK_END = {12: 21, 1: 21, 2: 21, 3: 22, 4: 22, 5: 22, 6: 23, 7: 23, 8: 23,
            9: 22, 10: 22, 11: 22}
RULES = ["plug-in", "random-delay", "after-peak", "latest", "valley-fill"]


def energy_from_project2(p2_src):
    """Recompute Wh/km at the battery with Project 2's code (if available)."""
    sys.path.insert(0, p2_src)
    import wmtc, vehicle   # noqa: E401  (Project 2 modules)
    v, tq = vehicle.follow(np.tile(wmtc.load("part1"), 3))
    p = tq * (v / 3.6) / vehicle.R_WHEEL
    return (p[p > 0] / 0.88).sum() / 3600 / (v.sum() / 3600)


def grid_kwh(profile, wh_per_km=WH_PER_KM_BATT, eta=ETA_CHARGER, factor=1.0):
    return PROFILES[profile]["km"] * wh_per_km * factor / 1000.0 / eta


def make_fleet(n_scooters, share_delivery, seed=1):
    rng = np.random.default_rng(seed)
    n_del = int(round(n_scooters * share_delivery))
    kinds = ["delivery"] * n_del + ["commuter"] * (n_scooters - n_del)
    fleet = []
    for k in kinds:
        a, b = PROFILES[k]["plug"]
        fleet.append(dict(kind=k, plug=rng.uniform(a, b), delay=rng.uniform(0, 600) / 60.0))
    return fleet


def charging_load(base_kw, index, fleet, rule, p_kw=P_CHARGER, factor=1.0):
    """Minute-by-minute charging load (kW) for the whole record.
    base_kw is the feeder load without scooters, used by valley-fill only."""
    n = len(index)
    load = np.zeros(n)
    base = np.array(base_kw, dtype=float)
    t0 = index[0]
    days = pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D")
    for day in days:
        for s in fleet:
            prof = PROFILES[s["kind"]]
            if day.weekday() not in prof["days"]:
                continue
            e_kwh = grid_kwh(s["kind"], factor=factor)
            need = int(np.ceil(e_kwh / p_kw * 60))           # minutes at full power
            plug = day + pd.Timedelta(hours=s["plug"])
            leave = day + pd.Timedelta(days=1, hours=prof["leave"])
            i_plug = int((plug - t0) / pd.Timedelta(minutes=1))
            i_leave = int((leave - t0) / pd.Timedelta(minutes=1))
            if i_leave > n:
                continue            # last night of the record: departure falls outside it
            if rule == "plug-in":
                start = i_plug
            elif rule == "random-delay":
                start = i_plug + int(round(s["delay"]))
            elif rule == "after-peak":
                pe = day + pd.Timedelta(hours=PEAK_END[day.month])
                start = max(i_plug, int((pe - t0) / pd.Timedelta(minutes=1)))
            elif rule == "latest":
                start = max(i_plug, i_leave - need)
            elif rule == "valley-fill":
                w = np.arange(i_plug, i_leave)
                pick = w[np.argsort(base[w], kind="stable")[:need]]
                load[pick] += p_kw
                base[pick] += p_kw                            # next scooter sees this one
                continue
            else:
                raise ValueError(rule)
            stop = min(start + need, n)
            load[start:stop] += p_kw
            base[start:stop] += p_kw
    return load


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.join(here, "..", "..", "e2w-controller-wearout", "src")
    if os.path.isdir(p2):
        print("Wh/km at battery from Project 2:", round(energy_from_project2(p2), 2))
    for k in PROFILES:
        print(k, "grid kWh/day", round(grid_kwh(k), 2),
              "hours at", P_CHARGER, "kW:", round(grid_kwh(k) / P_CHARGER, 2))
