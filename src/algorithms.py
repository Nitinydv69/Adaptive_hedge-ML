import numpy as np


def run_hedge(losses, eta):
    """Hedge with fixed learning rate eta. losses: (T, K) in [0, 1]."""
    T, K = losses.shape
    w = np.ones(K) / K
    cum = np.zeros(K)
    learner_losses = np.zeros(T)
    for t in range(T):
        learner_losses[t] = w @ losses[t]
        cum += losses[t]
        e = np.exp(-eta * (cum - cum.min()))   # shift for numerical stability
        w = e / e.sum()
    return learner_losses


def run_adahedge(losses, return_etas=False, return_gap=False):
    """
    AdaHedge (van Erven et al. 2011; de Rooij et al. 2014).
    eta_t = ln(K) / Delta_{t-1}, Delta = cumulative mixability gap.
    While Delta = 0 we have eta = infinity (follow the leader, uniform over ties).
    """
    T, K = losses.shape
    cum = np.zeros(K)
    learner_losses = np.zeros(T)
    etas = np.zeros(T)
    Delta = 0.0

    for t in range(T):
        # 1. weights
        if Delta <= 0.0:                       # eta = infinity
            eta = np.inf
            leaders = (cum == cum.min()).astype(float)
            w = leaders / leaders.sum()
        else:
            eta = np.log(K) / Delta
            e = np.exp(-eta * (cum - cum.min()))
            w = e / e.sum()
        etas[t] = eta

        # 2. incur loss
        l = losses[t]
        h = w @ l
        learner_losses[t] = h

        # 3. mixability gap  delta_t = h_t - m_t,  m_t = -(1/eta) ln sum_k w_k exp(-eta l_k)
        #    Shift by the min loss over the SUPPORT of w (not over all experts):
        #    exact for eta = inf (m_t = min loss on support) and avoids underflow.
        support = w > 0
        mn = l[support].min()
        if np.isinf(eta):
            m = mn
        else:
            m = mn - np.log(w[support] @ np.exp(-eta * (l[support] - mn))) / eta
        Delta += max(0.0, h - m)

        # 4. update cumulative losses
        cum += l

    if return_gap:
        return learner_losses, etas, Delta
    return (learner_losses, etas) if return_etas else learner_losses
