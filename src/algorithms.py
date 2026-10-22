import numpy as np


def run_hedge(losses, eta):
    """
    Standard Hedge algorithm with fixed learning rate eta.
    losses: shape (T, K)
    """
    T, K = losses.shape
    w = np.ones(K) / K
    cum_expert_losses = np.zeros(K)
    learner_losses = np.zeros(T)

    for t in range(T):
        loss_t = losses[t]
        learner_losses[t] = np.dot(w, loss_t)

        # Accumulate expert losses
        cum_expert_losses += loss_t

        # Exponential weights with numerical stability shift
        shifted = -eta * (cum_expert_losses - np.min(cum_expert_losses))
        exp_w = np.exp(shifted)
        w = exp_w / np.sum(exp_w)

    return learner_losses


def run_adahedge(losses):
    """
    AdaHedge (van Erven et al., NeurIPS 2011; de Rooij et al., JMLR 2014).
    Dynamically tunes eta_t = ln(K) / Delta_{t-1} using the cumulative mixability gap.
    """
    T, K = losses.shape
    w = np.ones(K) / K
    cum_expert_losses = np.zeros(K)
    learner_losses = np.zeros(T)
    cum_mix_gap = 0.0

    for t in range(T):
        # 1. Determine eta_t and compute weights from cumulative losses
        if cum_mix_gap <= 1e-12:
            eta_t = 1e6  # Approximates infinity for t=0 / zero gap
            if t == 0:
                w = np.ones(K) / K
            else:
                leaders = (cum_expert_losses == np.min(
                    cum_expert_losses)).astype(float)
                w = leaders / np.sum(leaders)
        else:
            eta_t = np.log(K) / cum_mix_gap
            shifted = -eta_t * (cum_expert_losses - np.min(cum_expert_losses))
            exp_w = np.exp(shifted)
            w = exp_w / np.sum(exp_w)

        # 2. Incur loss
        loss_t = losses[t]
        learner_loss_t = np.dot(w, loss_t)
        learner_losses[t] = learner_loss_t

        # 3. Compute mix loss and mixability gap delta_t
        min_ell = np.min(loss_t)
        shifted_ell = -eta_t * (loss_t - min_ell)
        exp_terms = np.exp(np.clip(shifted_ell, -700, 700))
        sum_exp = np.dot(w, exp_terms)
        mix_loss_t = min_ell - (1.0 / eta_t) * \
            np.log(np.maximum(sum_exp, 1e-300))

        delta_t = max(0.0, float(learner_loss_t - mix_loss_t))
        cum_mix_gap += delta_t

        # 4. Update cumulative losses
        cum_expert_losses += loss_t

    return learner_losses
