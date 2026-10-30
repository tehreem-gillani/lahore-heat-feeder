"""Full study: feeder load, scooter charging rules, transformer temperature and ageing.

Run after precon.py, feeder.py and tempsens.py. Writes results/results.json,
results/summary.txt and the figures.
"""
import json
import os

import numpy as np
import pandas as pd

import scooter
import tempsens
import transformer as tr
import weather

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "data", "derived")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

PF = 0.95                 # assumed power factor of the feeder load
S_RATED = 100.0           # kVA, base case
PV_KWP = 3.0              # kWp per house with rooftop PV (scenario only)
PV_PR = 0.8               # performance ratio, horizontal irradiance used as is
ADOPTION = [0.0, 0.25, 0.5, 0.75, 1.0]
SHARE_DELIVERY = 0.2
WARMUP = 2 * 1440         # minutes


def load_inputs():
    f = pd.read_csv(os.path.join(DERIVED, "feeder_1min.csv.gz"), parse_dates=["time"])
    idx = pd.DatetimeIndex(f["time"])
    w = weather.load().set_index("time")
    # POWER's hourly time stamp marks the start of the hour it averages
    # (POWER FAQ), so each value is placed at mid-hour before interpolating
    # to minutes; the first and last half hours take the nearest value.
    w.index = w.index + pd.Timedelta(minutes=30)
    w = w.reindex(w.index.union(idx)).interpolate("time").ffill().bfill().reindex(idx)
    houses = pd.read_csv(os.path.join(DERIVED, "feeder_houses.csv"))
    n_houses = int(houses["in_feeder"].sum())
    return idx, f["load_kw"].to_numpy(), w["t_air"].to_numpy(), w["ghi"].to_numpy(), n_houses


def feeder_temperature_fit(idx, load, t_air):
    s = pd.DataFrame({"p": load, "t": t_air}, index=idx).resample("D").mean()
    tr_mask = (s.index.isocalendar().week.to_numpy() % 2) == 0
    tb, base, slope = tempsens.fit_cp(s["t"].to_numpy()[tr_mask], s["p"].to_numpy()[tr_mask])
    yhat = tempsens.predict(s["t"].to_numpy()[~tr_mask], tb, base, slope)
    r2 = tempsens.r2(s["p"].to_numpy()[~tr_mask], yhat)
    return dict(t_balance=tb, base_kw=base, slope_kw_per_k=slope, r2_test=r2), s


def warmer(idx, load, t_air, fit, dT):
    """Load and ambient for a climate dT warmer, using the measured slope.
    Extra daily energy slope*dT on days above the balance temperature,
    spread over the day in proportion to that day's measured load."""
    s = pd.Series(load, index=idx)
    day_mean = s.groupby(idx.normalize()).transform("mean").to_numpy()
    day_t = pd.Series(t_air, index=idx).groupby(idx.normalize()).transform("mean").to_numpy()
    extra = np.where(day_t + dT > fit["t_balance"],
                     fit["slope_kw_per_k"] * (np.maximum(0, day_t + dT - fit["t_balance"])
                                              - np.maximum(0, day_t - fit["t_balance"])), 0.0)
    return load + extra * load / day_mean, t_air + dT


def metrics(idx, p_net, t_air, s_rated):
    K = np.abs(p_net) / PF / s_rated
    # warm-up: run the first two days twice before the record so the start
    # does not depend on an assumed initial temperature
    w = WARMUP
    _, th = tr.simulate(np.r_[K[:w], K], np.r_[t_air[:w], t_air])
    th = th[w:, 0]
    V = tr.ageing_rate(th)
    lol = float(V.sum() / 60.0)
    daily = pd.Series(V, index=idx).groupby(idx.normalize()).sum() / 60.0
    hs_day = pd.Series(th, index=idx).groupby(idx.normalize()).max()
    return dict(
        loss_of_life_h=round(lol, 1),
        loss_of_life_days=round(lol / 24, 2),
        peak_K=round(float(K.max()), 3),
        hs_max=round(float(th.max()), 1),
        hs_max_time=str(idx[int(th.argmax())]),
        hours_hs_above_120=round(float((th > tr.HS_LIMIT).sum() / 60), 1),
        hours_K_above_1=round(float((K > 1.0).sum() / 60), 1),
        worst_day=str(daily.idxmax().date()),
        worst_day_lol_h=round(float(daily.max()), 2),
        share_lol_jun_sep=round(float(daily[(daily.index.month >= 6) & (daily.index.month <= 9)].sum() / daily.sum()), 3),
        hs_days_above_120=int((hs_day > tr.HS_LIMIT).sum()),
    ), th


def run_case(idx, base, t_air, n_scooters, rule, s_rated=S_RATED, p_kw=scooter.P_CHARGER,
             factor=1.0, share_del=SHARE_DELIVERY, pv=None, seed=1):
    fleet = scooter.make_fleet(n_scooters, share_del, seed)
    ch = scooter.charging_load(base if pv is None else base - pv, idx, fleet, rule, p_kw, factor) \
        if n_scooters else np.zeros(len(idx))
    p = base + ch - (0 if pv is None else pv)
    m, th = metrics(idx, p, t_air, s_rated)
    m.update(n_scooters=n_scooters, rule=rule, s_rated=s_rated, charger_kw=p_kw,
             energy_factor=factor, share_delivery=share_del,
             charging_mwh=round(float(ch.sum() / 60 / 1000), 2))
    return m, ch, th


def main():
    os.makedirs(RES, exist_ok=True)
    idx, base, t_air, ghi, n_h = load_inputs()
    out = {"n_houses": n_h, "assumptions": dict(
        pf=PF, s_rated_kva=S_RATED, charger_kw=scooter.P_CHARGER, eta_charger=scooter.ETA_CHARGER,
        wh_per_km_batt=scooter.WH_PER_KM_BATT, share_delivery=SHARE_DELIVERY,
        commuter_kwh_day=round(scooter.grid_kwh("commuter"), 2),
        delivery_kwh_day=round(scooter.grid_kwh("delivery"), 2), pv_kwp=PV_KWP, pv_pr=PV_PR)}

    # feeder facts
    s = pd.Series(base, index=idx)
    hourly = s.resample("h").mean()
    prof = hourly.groupby([hourly.index.month, hourly.index.hour]).mean().unstack()
    out["feeder"] = dict(
        energy_mwh=round(float(base.sum() / 60 / 1000), 1),
        peak_1min_kw=round(float(base.max()), 1), peak_1min_time=str(idx[int(base.argmax())]),
        peak_hour_kw=round(float(hourly.max()), 1), peak_hour_time=str(hourly.idxmax()),
        jun_sep_mean_by_hour=prof.loc[6:9].mean().round(1).tolist(),
        dec_feb_mean_by_hour=prof.loc[[12, 1, 2]].mean().round(1).tolist(),
        hour_of_max_mean_by_month={int(m): int(prof.loc[m].idxmax()) for m in prof.index},
        hour_of_min_mean_by_month={int(m): int(prof.loc[m].idxmin()) for m in prof.index})
    fit, daily = feeder_temperature_fit(idx, base, t_air)
    out["feeder_temp_fit"] = {k: round(float(v), 3) for k, v in fit.items()}
    hs = tempsens.daily_table()
    tot = pd.read_csv(os.path.join(DERIVED, "temp_sensitivity_total.csv"))
    ac = pd.read_csv(os.path.join(DERIVED, "temp_sensitivity_ac_circuits.csv"))
    out["house_temp_fit"] = dict(
        n_total=len(tot), slope_median_w_per_k=float(tot.slope_w_per_k.median()),
        slope_iqr=[float(tot.slope_w_per_k.quantile(.25)), float(tot.slope_w_per_k.quantile(.75))],
        t_balance_median=float(tot.t_balance.median()), r2_test_median=float(tot.r2_test.median()),
        mae_test_median=float(tot.mae_test_kw.median()), mae_baseline_median=float(tot.mae_baseline_kw.median()),
        n_ac=len(ac), ac_slope_median_w_per_k=float(ac.slope_w_per_k.median()),
        ac_r2_test_median=float(ac.r2_test.median()), days_used=int(len(hs)))

    # A: baseline at two ratings
    out["A_baseline"] = {}
    th_store = {}
    for sr in (75.0, 100.0, 200.0):
        m, th = metrics(idx, base, t_air, sr)
        out["A_baseline"][str(int(sr))] = m
        th_store[f"base_{int(sr)}"] = th

    # B: adoption x rule, 100 kVA
    out["B_adoption"] = []
    ch_store = {}
    for a in ADOPTION:
        n = int(round(a * n_h))
        for rule in (scooter.RULES if n else ["plug-in"]):
            m, ch, th = run_case(idx, base, t_air, n, rule)
            m["adoption"] = a
            out["B_adoption"].append(m)
            if a == 1.0:
                ch_store[rule] = ch
                th_store[rule] = th
            print(a, rule, m["loss_of_life_h"], m["hs_max"], m["peak_K"])

    # C: sensitivity at full adoption, plug-in vs valley-fill (and latest)
    n = n_h
    pv = PV_KWP * PV_PR * ghi / 1000.0
    n_pv = int(round(0.3 * n_h))
    cases = {
        "base": {},
        "rating 75 kVA": dict(s_rated=75.0),
        "rating 200 kVA": dict(s_rated=200.0),
        "charger 0.35 kW": dict(p_kw=0.35),
        "charger 1.0 kW": dict(p_kw=1.0),
        "energy x1.5": dict(factor=1.5),
        "all commuters": dict(share_del=0.0),
        "half delivery": dict(share_del=0.5),
        "PV on 30% of houses": dict(pv=n_pv * pv),
        "other random seed": dict(seed=7),
    }
    out["C_sensitivity"] = []
    for name, kw in cases.items():
        for rule in ("plug-in", "after-peak", "latest", "valley-fill"):
            m, _, _ = run_case(idx, base, t_air, n, rule, **kw)
            m["case"] = name
            m.pop("hs_max_time", None)
            out["C_sensitivity"].append(m)
        # matching no-scooter reference for the same case
        kw0 = {k: v for k, v in kw.items() if k in ("s_rated", "pv")}
        m0, _, _ = run_case(idx, base, t_air, 0, "plug-in", **kw0)
        m0["case"] = name
        m0["rule"] = "no scooters"
        out["C_sensitivity"].append(m0)
        print(name, "done")

    # D: 2 K warmer, measured slope
    bw, tw = warmer(idx, base, t_air, fit, 2.0)
    out["D_warmer"] = []
    for nn, rule in ((0, "plug-in"), (n, "plug-in"), (n, "after-peak"), (n, "latest"), (n, "valley-fill")):
        m, _, _ = run_case(idx, bw, tw, nn, rule)
        m["case"] = "+2 K"
        out["D_warmer"].append(m)
    out["D_warmer_extra_energy_mwh"] = round(float((bw - base).sum() / 60 / 1000), 1)

    with open(os.path.join(RES, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    np.savez_compressed(os.path.join(RES, "series_for_figures.npz"),
                        time_min=np.arange(len(idx)), start=str(idx[0]), base=base, t_air=t_air,
                        **{f"ch_{k}": v for k, v in ch_store.items()},
                        **{f"th_{k}": v for k, v in th_store.items()})
    write_summary(out)


def write_summary(out):
    L = []
    f = out["feeder"]
    L.append(f"Houses on the virtual feeder: {out['n_houses']}")
    L.append(f"Feeder energy {f['energy_mwh']} MWh/yr; peak 1-min {f['peak_1min_kw']} kW at {f['peak_1min_time']}; "
             f"peak hour {f['peak_hour_kw']} kW at {f['peak_hour_time']}")
    L.append(f"Hour of highest mean load by month: {f['hour_of_max_mean_by_month']}")
    L.append(f"Hour of lowest mean load by month: {f['hour_of_min_mean_by_month']}")
    L.append(f"Jun-Sep mean by hour (kW): {f['jun_sep_mean_by_hour']}")
    h = out["house_temp_fit"]
    L.append(f"Houses: slope median {h['slope_median_w_per_k']:.0f} W/K (IQR {h['slope_iqr'][0]:.0f}-{h['slope_iqr'][1]:.0f}), "
             f"T_b median {h['t_balance_median']} C, test R2 median {h['r2_test_median']:.2f}, "
             f"MAE {h['mae_test_median']:.3f} vs {h['mae_baseline_median']:.3f} kW (n={h['n_total']}); "
             f"AC circuits slope median {h['ac_slope_median_w_per_k']:.0f} W/K, R2 {h['ac_r2_test_median']:.2f} (n={h['n_ac']})")
    L.append(f"Feeder fit: {out['feeder_temp_fit']}")
    L.append("A baseline:")
    for k, m in out["A_baseline"].items():
        L.append(f"  {k} kVA: {m}")
    L.append("B adoption (100 kVA):")
    for m in out["B_adoption"]:
        L.append(f"  adopt {m['adoption']:.2f} n={m['n_scooters']:2d} {m['rule']:12s} LoL {m['loss_of_life_h']:8.1f} h  "
                 f"hs max {m['hs_max']:6.1f}  peak K {m['peak_K']:.3f}  h>120 {m['hours_hs_above_120']:6.1f}  "
                 f"K>1 h {m['hours_K_above_1']:6.1f}  worst {m['worst_day']} ({m['worst_day_lol_h']} h)  JunSep {m['share_lol_jun_sep']}")
    L.append("C sensitivity (full adoption):")
    for m in out["C_sensitivity"]:
        L.append(f"  {m['case']:20s} {m['rule']:12s} LoL {m['loss_of_life_h']:8.1f} h  hs max {m['hs_max']:6.1f}  "
                 f"peak K {m['peak_K']:.3f}  h>120 {m['hours_hs_above_120']:6.1f}  charging {m['charging_mwh']} MWh")
    L.append(f"D +2 K (extra base energy {out['D_warmer_extra_energy_mwh']} MWh):")
    for m in out["D_warmer"]:
        L.append(f"  n={m['n_scooters']:2d} {m['rule']:12s} LoL {m['loss_of_life_h']:8.1f} h  hs max {m['hs_max']:6.1f}  "
                 f"peak K {m['peak_K']:.3f}  h>120 {m['hours_hs_above_120']:6.1f}")
    with open(os.path.join(RES, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
