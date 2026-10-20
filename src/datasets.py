import numpy as np


def generate_stochastic_losses(T=10000, K=10, best_p=0.3, other_p=0.5, seed=None):
    # bernaulli settings : One expert bernaulli = 0.3 and other bernaulli = 0.5
    if seed is not None:
        np.random.seed(seed)
    losses = np.random.binomial(1, other_p, size=(T, K)).astype(float)
    losses[:, 0] = np.random.binomial(1, best_p, size=T).astype(float)
    return losses

# Best expert Bernoulli(0.49), others Bernoulli(0.50)


def generate_low_gap_losses(T=10000, K=10, seed=None):
    return generate_stochastic_losses(T=T, K=K, best_p=0.49, other_p=0.50, seed=seed)


def generate_adversarial_losses(T=10000, K=10, num_shifts=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    losses = np.random.binomial(1, 0.5, size=(T, K)).astype(float)
    segment_len = T // num_shifts
    for s in range(num_shifts):
        start_idx = s * segment_len
        end_idx = (s + 1) * segment_len if s < num_shifts - 1 else T
        best_expert = s % K
        losses[start_idx:end_idx, best_expert] = np.random.binomial(
            1, 0.2, size=end_idx - start_idx).astype(float)
    return losses
