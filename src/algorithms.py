import numpy as np


def run_hedge(losses, eta):
    T, K = losses.shape
    w = np.ones(K)
    learner_losses = np.zeros(T)

    for t in range(T):
        p_t = w / np.sum(w)
        loss_t = losses[t]
        learner_losses[t] = np.dot(p_t, loss_t)
        w = w * np.exp(-eta * loss_t)

    return learner_losses


def run_adahedge(losses):
    T, K = losses.shape
    w = np.ones(K)
    learner_losses = np.zeros(T)
    cum_mix_gap = 0.0

    for t in range(T):
        eta_t = np.sqrt(np.log(K) / cum_mix_gap) if cum_mix_gap > 0 else 1.0
        p_t = w / np.sum(w)
        loss_t = losses[t]

        learner_loss_t = np.dot(p_t, loss_t)
        learner_losses[t] = learner_loss_t

        mix_loss_t = - (1.0 / eta_t) * \
            np.log(np.sum(p_t * np.exp(-eta_t * loss_t)))
        delta_t = learner_loss_t - mix_loss_t
        cum_mix_gap += delta_t

        w = p_t * np.exp(-eta_t * loss_t)
    return learner_losses
