"""Empirical check of the AdaHedge regret decomposition R_T <= 2 Delta_T
and of the worst-case order sqrt(T ln K). Writes results/bound_check.csv."""
import csv
import numpy as np
from src.algorithms import run_adahedge
from src.datasets import (generate_stochastic_losses, generate_low_gap_losses,
                          generate_adversarial_losses, generate_alternating_losses)

T, K, RUNS = 5000, 10, 20
gens = {"Stochastic": generate_stochastic_losses, "Low gap": generate_low_gap_losses,
        "Adversarial": generate_adversarial_losses, "Alternating": generate_alternating_losses}
rows = []
for n, g in gens.items():
    R, D, ok = [], [], 0
    for s in range(RUNS):
        L = g(T=T, K=K, seed=s)
        h, _, Delta = run_adahedge(L, return_gap=True)
        r = h.sum() - L.sum(0).min()
        R.append(r); D.append(Delta); ok += r <= 2 * Delta + 1e-9
    rows.append((n, np.mean(R), np.mean(D), 2 * np.mean(D), ok, np.sqrt(T * np.log(K))))
    print(n, "R=%.1f Delta=%.1f 2*Delta=%.1f holds %d/%d sqrt(TlnK)=%.1f" % (rows[-1][1], rows[-1][2], rows[-1][3], ok, RUNS, rows[-1][5]))
with open("results/bound_check.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["setting", "R_T", "Delta_T", "two_Delta_T", "holds_out_of_20", "sqrt_T_lnK"]); w.writerows(rows)
