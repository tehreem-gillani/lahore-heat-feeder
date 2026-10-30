"""Checks for the transformer model, gap detection and charging rules.
Run from the repo root:  python tests/test_models.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import precon       # noqa: E402
import scooter      # noqa: E402
import transformer as tr   # noqa: E402


def test_rated_steady_state():
    # IEC 60076-7: at rated load and 20 C ambient, top oil 75 C, hot spot 98 C, V = 1
    o, h = tr.simulate(np.ones(3000), np.full(3000, 20.0))
    assert abs(o[-1, 0] - 75.0) < 1e-6 and abs(h[-1, 0] - 98.0) < 1e-6
    assert abs(tr.ageing_rate(98.0) - 1.0) < 1e-12
    assert abs(tr.ageing_rate(104.0) - 2.0) < 1e-12        # doubles every 6 K


def test_filter_equals_loop():
    rng = np.random.default_rng(3)
    K = np.clip(0.8 + np.cumsum(rng.normal(0, 0.02, 5000)), 0, 2)
    ta = 30 + 8 * np.sin(np.arange(5000) / 300)
    o1, h1 = tr.simulate(K, ta)
    o2, h2 = tr.simulate_loop(K, ta)
    assert np.abs(o1[:, 0] - o2).max() < 1e-9 and np.abs(h1[:, 0] - h2).max() < 1e-9


def test_oil_time_constant():
    # after one oil time constant (180 min) the top-oil rise covers 1 - 1/e of a step,
    # within the error of the explicit 1-minute step
    n = 2000
    K = np.r_[np.zeros(10), np.ones(n)]
    o, _ = tr.simulate(K, np.zeros(len(K)))
    lo = 55.0 * (1 / 6) ** 0.8
    rise = (o[10 + 179, 0] - lo) / (55.0 - lo)
    assert abs(rise - (1 - np.exp(-1))) < 0.01, rise


def test_filled_mask():
    x = np.r_[np.random.default_rng(0).uniform(0.2, 2.0, 50), np.linspace(1, 2, 12),
              np.random.default_rng(1).uniform(0.2, 2.0, 50)]
    m = precon.filled_mask(x[:, None])
    assert m[50:62].all() and m.sum() == 12, m.sum()
    y = np.r_[np.random.default_rng(0).uniform(0.2, 2.0, 50), np.full(9, 1.0),
              np.random.default_rng(1).uniform(0.2, 2.0, 50)]
    assert precon.filled_mask(y[:, None]).sum() == 0          # 9 minutes: left as measured


def test_charging_energy_and_windows():
    idx = pd.date_range("2018-06-04", "2018-06-11 11:59", freq="min")   # Mon to next Mon noon
    base = 40 + 20 * np.cos(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
    fleet = scooter.make_fleet(10, 0.3, seed=2)
    expect = 0.0
    for d in pd.date_range("2018-06-04", "2018-06-10"):
        for s in fleet:
            if d.weekday() in scooter.PROFILES[s["kind"]]["days"]:
                need = int(np.ceil(scooter.grid_kwh(s["kind"]) / scooter.P_CHARGER * 60))
                expect += need * scooter.P_CHARGER / 60
    for rule in scooter.RULES:
        ch = scooter.charging_load(base, idx, fleet, rule)
        got = ch.sum() / 60
        assert abs(got - expect) < 1e-6, (rule, got, expect)
        # nobody charges between 11:00 and 17:59: all scooters are out
        h = idx.hour
        assert ch[(h >= 11) & (h < 18)].sum() == 0, rule


def test_after_peak_start():
    idx = pd.date_range("2018-06-04", "2018-06-05 23:59", freq="min")
    fleet = [dict(kind="commuter", plug=18.2, delay=3.0)]
    ch = scooter.charging_load(np.zeros(len(idx)), idx, fleet, "after-peak")
    first = idx[np.argmax(ch > 0)]
    assert first == pd.Timestamp("2018-06-04 23:00"), first


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
