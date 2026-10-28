import numpy as np


def generate_stochastic_losses(T=10000, K=10, best_p=0.3, other_p=0.5, seed=None):
    """Expert 0 ~ Bernoulli(best_p); all others ~ Bernoulli(other_p), i.i.d. over time."""
    rng = np.random.default_rng(seed)
    losses = rng.binomial(1, other_p, size=(T, K)).astype(float)
    losses[:, 0] = rng.binomial(1, best_p, size=T)
    return losses


def generate_low_gap_losses(T=10000, K=10, seed=None):
    """Best expert Bernoulli(0.49), others Bernoulli(0.50): gap of only 0.01."""
    return generate_stochastic_losses(T=T, K=K, best_p=0.49, other_p=0.50, seed=seed)


def generate_adversarial_losses(T=10000, K=10, num_shifts=3, seed=None):
    """Piecewise-stationary: in segment s, expert s is Bernoulli(0.2), others Bernoulli(0.5)."""
    rng = np.random.default_rng(seed)
    losses = rng.binomial(1, 0.5, size=(T, K)).astype(float)
    seg = T // num_shifts
    for s in range(num_shifts):
        a = s * seg
        b = (s + 1) * seg if s < num_shifts - 1 else T
        losses[a:b, s % K] = rng.binomial(1, 0.2, size=b - a)
    return losses


def generate_alternating_losses(T=10000, K=10, seed=None):
    """
    Deterministic worst case for Follow-the-Leader (extra setting).
    Experts 0,1 alternate between loss 0 and 1 (out of phase); the others always lose 1.
    The leader flips every round, so a greedy learner loses 1 each round while the
    best expert loses ~T/2.  `seed` is unused (kept for a uniform interface).
    """
    losses = np.ones((T, K))
    losses[0, 0] = losses[0, 1] = 0.5
    t = np.arange(1, T)
    losses[1:, 0] = (t % 2 == 0).astype(float)
    losses[1:, 1] = (t % 2 == 1).astype(float)
    return losses
