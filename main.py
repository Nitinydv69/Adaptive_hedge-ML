import os
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import (
    generate_stochastic_losses,
    generate_low_gap_losses,
    generate_adversarial_losses,
)
from src.algorithms import run_hedge, run_adahedge


def run_experiment_suite(setting_name, generator_func, T=5000, K=10, num_runs=20):
    print(f"Running experiments for: {setting_name}...")
    eta_opt = np.sqrt(8 * np.log(K) / T)
    etas = [0.005, eta_opt, 0.5]
    eta_labels = [
        r"Hedge ($\eta=0.005$, small)",
        rf"Hedge ($\eta^*={eta_opt:.3f}$, optimal)",
        r"Hedge ($\eta=0.5$, large)"
    ]

    regrets = {
        "AdaHedge": np.zeros((num_runs, T)),
        eta_labels[0]: np.zeros((num_runs, T)),
        eta_labels[1]: np.zeros((num_runs, T)),
        eta_labels[2]: np.zeros((num_runs, T)),
    }

    cum_losses = {
        "Best Single Expert": np.zeros((num_runs, T)),
        "AdaHedge": np.zeros((num_runs, T)),
        eta_labels[0]: np.zeros((num_runs, T)),
        eta_labels[1]: np.zeros((num_runs, T)),
        eta_labels[2]: np.zeros((num_runs, T)),
    }

    for seed in range(num_runs):
        losses = generator_func(T=T, K=K, seed=seed)
        cum_exp_losses = np.cumsum(losses, axis=0)
        best_exp = np.min(cum_exp_losses, axis=1)
        cum_losses["Best Single Expert"][seed] = best_exp

        # AdaHedge
        a_loss = run_adahedge(losses)
        cum_a = np.cumsum(a_loss)
        regrets["AdaHedge"][seed] = cum_a - best_exp
        cum_losses["AdaHedge"][seed] = cum_a

        # Fixed Hedge variants
        for idx, eta in enumerate(etas):
            h_loss = run_hedge(losses, eta=eta)
            cum_h = np.cumsum(h_loss)
            regrets[eta_labels[idx]][seed] = cum_h - best_exp
            cum_losses[eta_labels[idx]][seed] = cum_h

    os.makedirs("plots", exist_ok=True)
    time_axis = np.arange(1, T + 1)
    file_prefix = setting_name.lower().replace(" ", "_")

    # 1. Regret Plot
    plt.figure(figsize=(8, 4.5))
    for label, reg_matrix in regrets.items():
        mean_reg = np.mean(reg_matrix, axis=0)
        std_reg = np.std(reg_matrix, axis=0)
        plt.plot(time_axis, mean_reg, label=label, linewidth=2.0)
        plt.fill_between(time_axis, mean_reg - std_reg,
                         mean_reg + std_reg, alpha=0.15)

    plt.title(
        f"Cumulative Regret vs. Time — {setting_name}", fontweight='bold')
    plt.xlabel("Time Step (t)")
    plt.ylabel("Cumulative Regret")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/regret_{file_prefix}.png", dpi=300)
    plt.close()

    # 2. Cumulative Loss Plot
    plt.figure(figsize=(8, 4.5))
    for label, loss_matrix in cum_losses.items():
        mean_l = np.mean(loss_matrix, axis=0)
        ls = ":" if label == "Best Single Expert" else "-"
        plt.plot(time_axis, mean_l, label=label, linestyle=ls, linewidth=2.0)

    plt.title(f"Cumulative Loss vs. Time — {setting_name}", fontweight='bold')
    plt.xlabel("Time Step (t)")
    plt.ylabel("Cumulative Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/cum_loss_{file_prefix}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_experiment_suite("Stochastic", generate_stochastic_losses)
    run_experiment_suite("Low Gap Stochastic", generate_low_gap_losses)
    run_experiment_suite("Adversarial", generate_adversarial_losses)
    print("All plots generated successfully in /plots directory!")
