"""Figures for the report and README (run after run_all.py)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tempsens
import weather

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
DERIVED = os.path.join(HERE, "..", "data", "derived")

# colours: categorical slots 1-4 of the reference palette, gray for "no scooters"
C = {"plug-in": "#2a78d6", "after-peak": "#eb6834", "latest": "#1baf7a",
     "valley-fill": "#6250d6", "none": "#6f6e69", "winter": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"plug-in": "charge on plug-in", "after-peak": "timer: start after NEPRA peak",
         "latest": "timer: finish at departure", "valley-fill": "ideal valley filling",
         "random-delay": "plug-in + random delay (max 600 s)"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 200, "figure.dpi": 100})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


def f1_daily_shape(out):
    f = out["feeder"]
    h = np.arange(24)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axvspan(19, 23, color="#f3d9cc", alpha=0.6, lw=0)
    ax.text(21, 4, "NEPRA peak\nJun-Aug", ha="center", va="bottom", color=INK2, fontsize=8)
    ax.plot(h, f["jun_sep_mean_by_hour"], color=C["plug-in"], marker="o", ms=3, label="June to September")
    ax.plot(h, f["dec_feb_mean_by_hour"], color=C["winter"], marker="o", ms=3, label="December to February")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.3, 23.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("hour of day (Pakistan time); value = mean over that hour")
    ax.set_ylabel("feeder load (kW)")
    ax.legend(loc="upper center", ncol=2)
    save(fig, "f1_daily_shape.png")


def f2_temperature(out):
    idx = None
    fl = pd.read_csv(os.path.join(DERIVED, "feeder_1min.csv.gz"), parse_dates=["time"]).set_index("time")
    w = weather.load().set_index("time")
    d = pd.concat([fl["load_kw"].resample("D").mean(), w["t_air"].resample("D").mean()], axis=1).dropna()
    train = (d.index.isocalendar().week.to_numpy() % 2) == 0
    fit = out["feeder_temp_fit"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.scatter(d["t_air"][train], d["load_kw"][train], s=10, color=C["plug-in"], alpha=0.6,
               edgecolor="white", linewidth=0.4, label="fitting days (even weeks)")
    ax.scatter(d["t_air"][~train], d["load_kw"][~train], s=10, color=C["after-peak"], alpha=0.6,
               edgecolor="white", linewidth=0.4, label="test days (odd weeks)")
    T = np.linspace(d["t_air"].min(), d["t_air"].max(), 100)
    ax.plot(T, tempsens.predict(T, fit["t_balance"], fit["base_kw"], fit["slope_kw_per_k"]),
            color=INK, lw=1.5, label=f"fit: {fit['slope_kw_per_k']:.1f} kW per K above {fit['t_balance']:.1f} °C")
    ax.set_xlabel("daily mean air temperature, Lahore (°C)")
    ax.set_ylabel("daily mean feeder load (kW)")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "f2_temperature.png")


def f3_week(out):
    z = np.load(os.path.join(RES, "series_for_figures.npz"))
    t = pd.Timestamp(str(z["start"])) + pd.to_timedelta(z["time_min"], unit="min")
    sel = (t >= "2018-06-03") & (t < "2018-06-07")
    base = z["base"][sel]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
    sm = lambda x: pd.Series(x).rolling(15, center=True, min_periods=1).mean().to_numpy()
    for r in ("plug-in", "after-peak", "latest"):
        axs[0].plot(t[sel], sm(base + z[f"ch_{r}"][sel]), color=C[r], lw=1.2, label=LABEL[r])
    axs[0].plot(t[sel], sm(base), color=INK, lw=0.8, label="no scooters")
    axs[0].axhline(100 * 0.95, color=INK2, lw=0.8, ls="--")
    axs[0].text(t[sel][10], 97, "100 kVA rating at pf 0.95", color=INK2, fontsize=7.5)
    axs[0].set_ylabel("feeder load (kW),\n15-min mean")
    axs[0].legend(loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.32))
    for r in ("plug-in", "after-peak", "latest"):
        axs[1].plot(t[sel], z[f"th_{r}"][sel], color=C[r], lw=1.2)
    axs[1].plot(t[sel], z["th_base_100"][sel], color=INK, lw=0.8)
    axs[1].plot(t[sel], z["t_air"][sel], color=INK2, lw=1, ls=":")
    axs[1].text(t[sel][60], z["t_air"][sel][60] + 3, "air temperature", color=INK2, fontsize=7.5)
    axs[1].set_ylabel("hot-spot temp (°C)")
    axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d Jun\n%H:%M"))
    save(fig, "f3_hot_days.png")


def f4_adoption(out):
    B = pd.DataFrame(out["B_adoption"])
    base = B[(B.n_scooters == 0)]["loss_of_life_h"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for r in ("after-peak", "plug-in", "latest", "valley-fill"):
        s = pd.concat([B[B.n_scooters == 0].assign(rule=r), B[B.rule == r]]).sort_values("adoption")
        y = s["loss_of_life_h"].to_numpy() / base
        ax.plot(s["adoption"] * 100, y, color=C[r], marker="o", ms=4, label=LABEL[r])
        dy = {"latest": 0.008, "valley-fill": -0.008}.get(r, 0.0)
        ax.text(102, y[-1] + dy, f"{y[-1]:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("houses with an e-scooter (%)")
    ax.set_ylabel("yearly insulation ageing\n(relative to no scooters)")
    ax.set_xlim(-2, 110)
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "f4_adoption.png")


def f5_sensitivity(out):
    C_ = pd.DataFrame(out["C_sensitivity"])
    cases = ["base", "rating 75 kVA", "rating 200 kVA", "charger 1.0 kW", "energy x1.5",
             "all commuters", "half delivery", "PV on 30% of houses"]
    rules = ["after-peak", "plug-in", "latest"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    hgt = 0.26
    for j, r in enumerate(rules):
        vals = []
        for c in cases:
            ref = C_[(C_.case == c) & (C_.rule == "no scooters")]["loss_of_life_h"].iloc[0]
            vals.append(C_[(C_.case == c) & (C_.rule == r)]["loss_of_life_h"].iloc[0] / ref - 1)
        y = np.arange(len(cases)) + (j - 1) * hgt
        ax.barh(y, np.array(vals) * 100, height=hgt - 0.03, color=C[r], label=LABEL[r])
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases)
    ax.invert_yaxis()
    ax.set_xlabel("extra yearly ageing from 34 scooters (%)\n(each case against its own no-scooter run)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    save(fig, "f5_sensitivity.png")


def main():
    out = json.load(open(os.path.join(RES, "results.json")))
    f1_daily_shape(out)
    f2_temperature(out)
    f3_week(out)
    f4_adoption(out)
    f5_sensitivity(out)


if __name__ == "__main__":
    main()
