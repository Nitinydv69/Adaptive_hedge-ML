"""Extra analyses for the report (all use seeds 0-19, same sequences as main.py).

1. Paired differences AdaHedge - Hedge (95% CI)          -> results/paired.csv
2. Low-gap pseudo-regret (vs the best-in-expectation expert) -> results/pseudo_regret.csv
3. Scaling with the horizon T (500, 5000, 50000)          -> results/scaling.csv, plots/scaling.png
4. eta-sweep with the Hedge worst-case bound overlaid     -> plots/eta_sweep_theory.png
5. Log-log slope of AdaHedge's eta_t                      -> results/eta_slopes.csv
6. Cumulative loss with zoom (readable version)           -> plots/cum_loss_zoom.png
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import (generate_stochastic_losses, generate_low_gap_losses,
                          generate_adversarial_losses, generate_alternating_losses)
from src.algorithms import run_hedge, run_adahedge

K, NUM_RUNS = 10, 20
T_MAIN = 5000
T_CRIT = 2.093          # t quantile (97.5%) for 19 degrees of freedom (20 seeds)
SWEEP_ETAS = np.logspace(-2.5, 0.5, 11)
SETTINGS = {"Stochastic": generate_stochastic_losses,
            "Low gap": generate_low_gap_losses,
            "Adversarial": generate_adversarial_losses,
            "Alternating": generate_alternating_losses}
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


def final_regret(learner_losses, L):
    return learner_losses.sum() - L.sum(0).min()


def ci95(x):
    x = np.asarray(x)
    half = T_CRIT * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), half


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------- 1, 2, 4, 5
eta_star = np.sqrt(8 * np.log(K) / T_MAIN)
paired_rows, pseudo_rows, slope_rows = [], [], []
sweep_all, ada_all = {}, {}

for name, gen in SETTINGS.items():
    print("main-T analyses:", name, flush=True)
    ada, h_star, h_half, sweep = [], [], [], np.zeros((NUM_RUNS, len(SWEEP_ETAS)))
    pseudo = {"AdaHedge": [], "Hedge eta*": [], "Hedge eta=0.5": [], "Uniform (no learning)": []}
    etas_traj = []
    for s in range(NUM_RUNS):
        L = gen(T=T_MAIN, K=K, seed=s)
        a, et = run_adahedge(L, return_etas=True)
        etas_traj.append(et)
        hs, hh = run_hedge(L, eta_star), run_hedge(L, 0.5)
        ada.append(final_regret(a, L)); h_star.append(final_regret(hs, L)); h_half.append(final_regret(hh, L))
        for j, e in enumerate(SWEEP_ETAS):
            sweep[s, j] = final_regret(run_hedge(L, e), L)
        # pseudo-regret against expert 0 (best in expectation for the stochastic-type settings)
        pseudo["AdaHedge"].append(a.sum() - L[:, 0].sum())
        pseudo["Hedge eta*"].append(hs.sum() - L[:, 0].sum())
        pseudo["Hedge eta=0.5"].append(hh.sum() - L[:, 0].sum())
        pseudo["Uniform (no learning)"].append(L.mean(1).sum() - L[:, 0].sum())
    ada, h_star, h_half = map(np.array, (ada, h_star, h_half))
    best_j = sweep.mean(0).argmin()
    oracle = sweep[:, best_j]
    for label, ref in [("Hedge eta*", h_star), ("Hedge eta=0.5", h_half), ("Hedge oracle eta", oracle)]:
        m, h = ci95(ada - ref)
        paired_rows.append((name, label, m, h))
    if name in ("Stochastic", "Low gap"):
        for label, v in pseudo.items():
            m, h = ci95(v)
            pseudo_rows.append((name, label, m, h))
    sweep_all[name] = sweep
    ada_all[name] = ada
    # slope of the median eta_t on log-log scale, t in [500, 5000]
    med = np.median(np.array(etas_traj), axis=0)
    t = np.arange(1, T_MAIN + 1)
    sel = (t >= 500)
    slope = np.polyfit(np.log(t[sel]), np.log(med[sel]), 1)[0]
    slope_rows.append((name, slope))

write_csv("results/paired.csv", ["setting", "comparison", "mean_diff", "ci95_halfwidth"], paired_rows)
write_csv("results/pseudo_regret.csv", ["setting", "algorithm", "mean", "ci95_halfwidth"], pseudo_rows)
write_csv("results/eta_slopes.csv", ["setting", "loglog_slope_t_ge_500"], slope_rows)

# eta sweep + theory
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
for ax, (name, sw) in zip(axes.ravel(), sweep_all.items()):
    m, sd = sw.mean(0), sw.std(0)
    ax.loglog(SWEEP_ETAS, m, "o-", label="Hedge (observed)")
    ax.fill_between(SWEEP_ETAS, np.maximum(m - sd, 1e-1), m + sd, alpha=.15)
    fine = np.logspace(-2.5, 0.5, 200)
    ax.loglog(fine, np.log(K) / fine + fine * T_MAIN / 8, "k--", lw=1, label=r"bound $\ln K/\eta+\eta T/8$")
    a = ada_all[name]
    ax.axhline(a.mean(), color="C3", lw=2, label="AdaHedge")
    ax.set_title(name); ax.set_xlabel(r"$\eta$"); ax.set_ylabel(f"Regret at $T={T_MAIN}$")
    ax.grid(ls="--", alpha=.5, which="both")
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("plots/eta_sweep_theory.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 3 scaling
print("scaling in T ...", flush=True)
Ts = [500, 5000, 50000]
scal_rows = []
scal = {}
for name in ("Stochastic", "Alternating"):
    gen = SETTINGS[name]
    for T in Ts:
        runs = NUM_RUNS if name == "Stochastic" else 1       # alternating is deterministic
        if name == "Stochastic" and T == 50000:
            runs = 10                                        # keep the runtime reasonable
        res = {"AdaHedge": [], "Hedge eta=0.5": [], "Hedge eta*(T)": []}
        for s in range(runs):
            L = gen(T=T, K=K, seed=s)
            res["AdaHedge"].append(final_regret(run_adahedge(L), L))
            res["Hedge eta=0.5"].append(final_regret(run_hedge(L, 0.5), L))
            res["Hedge eta*(T)"].append(final_regret(run_hedge(L, np.sqrt(8 * np.log(K) / T)), L))
        for alg, v in res.items():
            scal[(name, alg, T)] = np.mean(v)
            scal_rows.append((name, alg, T, runs, np.mean(v), np.std(v)))
        print("  ", name, T, flush=True)
write_csv("results/scaling.csv", ["setting", "algorithm", "T", "runs", "regret_mean", "regret_std"], scal_rows)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, name in zip(axes, ("Stochastic", "Alternating")):
    for alg in ("AdaHedge", "Hedge eta=0.5", "Hedge eta*(T)"):
        ax.loglog(Ts, [scal[(name, alg, T)] for T in Ts], "o-", lw=2, label=alg)
    ax.loglog(Ts, [np.sqrt(T * np.log(K) / 2) for T in Ts], "k--", lw=1, label=r"$\sqrt{(T/2)\ln K}$")
    ax.loglog(Ts, [0.06 * T for T in Ts], ":", color="gray", lw=1, label=r"linear in $T$ (slope 1)")
    ax.set_title(name); ax.set_xlabel("Horizon $T$"); ax.set_ylabel("Final regret")
    ax.grid(ls="--", alpha=.5, which="both")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("plots/scaling.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 6 zoomed cumulative loss
print("zoomed cumulative loss ...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
t = np.arange(1, T_MAIN + 1)
for col, name in enumerate(("Stochastic", "Alternating")):
    gen = SETTINGS[name]
    curves = {"Best expert": [], "AdaHedge": [], r"Hedge $\eta^*$": [], r"Hedge $\eta=0.5$": []}
    for s in range(NUM_RUNS if name == "Stochastic" else 1):
        L = gen(T=T_MAIN, K=K, seed=s)
        curves["Best expert"].append(np.cumsum(L, 0).min(1))
        curves["AdaHedge"].append(np.cumsum(run_adahedge(L)))
        curves[r"Hedge $\eta^*$"].append(np.cumsum(run_hedge(L, eta_star)))
        curves[r"Hedge $\eta=0.5$"].append(np.cumsum(run_hedge(L, 0.5)))
    for lab, v in curves.items():
        mean = np.mean(v, 0)
        for row in (0, 1):
            axes[row, col].plot(t, mean, ":" if lab == "Best expert" else "-", lw=2, label=lab)
    zoom = t >= 4500
    lo = min(np.mean(v, 0)[zoom].min() for v in curves.values())
    hi = max(np.mean(v, 0)[zoom].max() for v in curves.values())
    axes[1, col].set_xlim(4500, 5000); axes[1, col].set_ylim(lo - 5, hi + 5)
    axes[0, col].set_title(f"{name}: cumulative loss")
    axes[1, col].set_title("zoom on the last 500 rounds")
    for row in (0, 1):
        axes[row, col].set_xlabel("Round $t$"); axes[row, col].set_ylabel("Cumulative loss")
        axes[row, col].grid(ls="--", alpha=.5)
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("plots/cum_loss_zoom.png", dpi=200); plt.close()
print("Done.")
