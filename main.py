import os
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import (
    generate_stochastic_losses,
    generate_low_gap_losses,
    generate_adversarial_losses,
)
from src.algorithms import run_hedge, run_adahedge


def compute_regret(learner_losses, losses):
    """Compute regret vs. best fixed expert in hindsight."""
    cum_learner_loss = np.cumsum(learner_losses)
    cum_expert_losses = np.cumsum(losses, axis=0)
    best_expert_loss = np.min(cum_expert_losses, axis=1)
    return cum_learner_loss - best_expert_loss


def run_experiment_suite(setting_name, generator_func, T=10000, K=10, num_runs=30):
    print(f"Running experiments for: {setting_name}...")
    etas = [0.01, np.sqrt(8 * np.log(K) / T), 0.5]

    regrets = {
        "AdaHedge": np.zeros((num_runs, T)),
        f"Hedge (eta={etas[0]})": np.zeros((num_runs, T)),
        f"Hedge (optimal eta)": np.zeros((num_runs, T)),
        f"Hedge (eta={etas[2]})": np.zeros((num_runs, T)),
    }

    for seed in range(num_runs):
        losses = generator_func(T=T, K=K, seed=seed)

        # AdaHedge
        a_loss = run_adahedge(losses)
        regrets["AdaHedge"][seed] = compute_regret(a_loss, losses)

        # Fixed Hedge variants
        h0_loss = run_hedge(losses, eta=etas[0])
        regrets[f"Hedge (eta={etas[0]})"][seed] = compute_regret(
            h0_loss, losses)

        h1_loss = run_hedge(losses, eta=etas[1])
        regrets[f"Hedge (optimal eta)"][seed] = compute_regret(h1_loss, losses)

        h2_loss = run_hedge(losses, eta=etas[2])
            h2_loss, losses)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(9, 5))
    time_axis = np.arange(1, T + 1)

    for label, reg_matrix in regrets.items():
        mean_reg = np.mean(reg_matrix, axis=0)
        std_reg = np.std(reg_matrix, axis=0)
        plt.plot(time_axis, mean_reg, label=label)
        plt.fill_between(time_axis, mean_reg - std_reg,
                         mean_reg + std_reg, alpha=0.15)

    plt.title(f"Regret vs. Time — {setting_name}")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Cumulative Regret")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"plots/regret_{setting_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_experiment_suite("Stochastic", generate_stochastic_losses)
    run_experiment_suite("Low Gap Stochastic", generate_low_gap_losses)
    run_experiment_suite("Adversarial", generate_adversarial_losses)
    print("All plots generated successfully in /plots directory!")
