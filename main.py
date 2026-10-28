import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import (generate_stochastic_losses, generate_low_gap_losses,
                          generate_adversarial_losses, generate_alternating_losses)
from src.algorithms import run_hedge, run_adahedge

T, K, NUM_RUNS = 5000, 10, 20
SWEEP_ETAS = np.logspace(-2.5, 0.5, 11)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


def run_experiment_suite(name, gen):
    print(f"Running: {name} ...")
    eta_opt = np.sqrt(8 * np.log(K) / T)
    fixed = {rf"Hedge $\eta=0.005$ (small)": 0.005,
             rf"Hedge $\eta^*={eta_opt:.3f}$ (theory-optimal)": eta_opt,
             rf"Hedge $\eta=0.5$ (large)": 0.5}
    names = ["AdaHedge"] + list(fixed)
    regret = {n: np.zeros((NUM_RUNS, T)) for n in names}
    cumloss = {n: np.zeros((NUM_RUNS, T)) for n in names + ["Best expert"]}
    sweep = np.zeros((NUM_RUNS, len(SWEEP_ETAS)))
    eta_traj = np.zeros((NUM_RUNS, T))

    for s in range(NUM_RUNS):
        L = gen(T=T, K=K, seed=s)                    # same sequence for every algorithm
        best = np.cumsum(L, axis=0).min(axis=1)      # best fixed expert in hindsight at each t
        cumloss["Best expert"][s] = best
        a, et = run_adahedge(L, return_etas=True)
        eta_traj[s] = et
        runs = {"AdaHedge": a}
        for n, e in fixed.items():
            runs[n] = run_hedge(L, e)
        for n, l in runs.items():
            c = np.cumsum(l)
            cumloss[n][s] = c
            regret[n][s] = c - best
        for j, e in enumerate(SWEEP_ETAS):
            sweep[s, j] = np.sum(run_hedge(L, e)) - best[-1]

    tag = name.lower().replace(" ", "_").replace("-", "_")
    t = np.arange(1, T + 1)

    # --- regret vs time
    plt.figure(figsize=(8, 4.5))
    for n in names:
        m, sd = regret[n].mean(0), regret[n].std(0)
        line, = plt.plot(t, m, lw=2, label=n)
        plt.fill_between(t, m - sd, m + sd, alpha=0.15, color=line.get_color())
    plt.plot(t, np.sqrt(t * np.log(K) / 2), "k--", lw=1, label=r"worst-case bound $\sqrt{(t/2)\ln K}$")
    plt.axhline(0, color="gray", lw=0.6)
    plt.title(f"Cumulative regret vs. time: {name}", fontweight="bold")
    plt.xlabel("Round $t$"); plt.ylabel("Regret"); plt.grid(ls="--", alpha=.5)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"plots/regret_{tag}.png", dpi=200); plt.close()

    # --- cumulative loss vs time
    plt.figure(figsize=(8, 4.5))
    for n, M in cumloss.items():
        plt.plot(t, M.mean(0), ":" if n == "Best expert" else "-", lw=2, label=n)
    plt.title(f"Cumulative loss vs. time: {name}", fontweight="bold")
    plt.xlabel("Round $t$"); plt.ylabel("Cumulative loss"); plt.grid(ls="--", alpha=.5)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"plots/cum_loss_{tag}.png", dpi=200); plt.close()

    # --- final regret vs eta (Hedge sweep) with AdaHedge reference
    plt.figure(figsize=(6, 4))
    m, sd = sweep.mean(0), sweep.std(0)
    plt.semilogx(SWEEP_ETAS, m, "o-", label="Hedge, fixed $\\eta$")
    plt.fill_between(SWEEP_ETAS, m - sd, m + sd, alpha=.15)
    af = regret["AdaHedge"][:, -1]
    plt.axhline(af.mean(), color="C3", lw=2, label="AdaHedge (no tuning)")
    plt.axhspan(af.mean() - af.std(), af.mean() + af.std(), color="C3", alpha=.12)
    plt.title(f"Final regret vs. $\\eta$: {name}", fontweight="bold")
    plt.xlabel("$\\eta$"); plt.ylabel(f"Regret at $T={T}$"); plt.grid(ls="--", alpha=.5)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"plots/eta_sweep_{tag}.png", dpi=200); plt.close()

    # summary rows
    rows = [(name, n, regret[n][:, -1].mean(), regret[n][:, -1].std()) for n in names]
    rows.append((name, "Hedge best eta in sweep (oracle)", sweep.mean(0).min(),
                 sweep.std(0)[sweep.mean(0).argmin()]))
    return rows, eta_traj


if __name__ == "__main__":
    settings = {"Stochastic": generate_stochastic_losses,
                "Low gap": generate_low_gap_losses,
                "Adversarial": generate_adversarial_losses,
                "Alternating": generate_alternating_losses}
    all_rows, trajs = [], {}
    for n, g in settings.items():
        r, tr = run_experiment_suite(n, g)
        all_rows += r; trajs[n] = tr

    with open("results/summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting", "algorithm", "final_regret_mean", "final_regret_std"])
        w.writerows(all_rows)

    # eta_t trajectories of AdaHedge (skip t=1 where eta = inf)
    plt.figure(figsize=(7, 4))
    t = np.arange(1, T + 1)
    for n, tr in trajs.items():
        plt.loglog(t[1:], np.median(tr[:, 1:], axis=0), lw=2, label=n)
    plt.title(r"AdaHedge learning rate $\eta_t$ (median over runs)", fontweight="bold")
    plt.xlabel("Round $t$"); plt.ylabel(r"$\eta_t$"); plt.grid(ls="--", alpha=.5, which="both")
    plt.legend(); plt.tight_layout(); plt.savefig("plots/adahedge_eta.png", dpi=200); plt.close()
    print("Done.")
