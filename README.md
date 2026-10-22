# Online Learning with Adaptive Hedge

**Course:** Statistical Methods for Machine Learning (A.Y. 2025/26)  
**Instructor:** Prof. Nicolò Cesa-Bianchi  
**Author:** Nitin Yadav (University of Milan)  

---

## Abstract
This repository implements and evaluates **Adaptive Hedge (AdaHedge)** against the classical **Hedge** algorithm in the framework of prediction with expert advice. While classical Hedge achieves minimax optimal regret $\mathcal{O}(\sqrt{T \ln K})$, it requires pre-tuning a static learning rate $\eta$ using prior knowledge of the time horizon $T$, and is overly pessimistic in benign stochastic environments. AdaHedge (*van Erven et al., NeurIPS 2011; de Rooij et al., JMLR 2014*) adaptively tunes the learning rate online using the cumulative mixability gap $\Delta_t = \sum_{s=1}^t \delta_s$.

We implement both algorithms from scratch in Python and compare their performance across three synthetic regimes:
1. **Stochastic Regime:** Clear best expert ($\text{Bernoulli}(0.3)$ vs. $\text{Bernoulli}(0.5)$).
2. **Adversarial Regime:** Optimal expert identity shifts across temporal phases.
3. **Low-Gap Regime:** Subtle performance margin ($\text{Bernoulli}(0.49)$ vs. $\text{Bernoulli}(0.50)$).

---

## Repository Structure

```
.
├── README.md               # Project documentation and reproduction guide
├── report.pdf              # Final 6-page academic report (submission PDF)
├── report.tex              # Clean academic LaTeX source of the report
├── requirements.txt        # Python package dependencies (numpy, matplotlib)
├── main.py                 # Main experiment runner generating plots and metrics
├── src/
│   ├── algorithms.py       # From-scratch implementations of Hedge and AdaHedge
│   └── datasets.py         # Synthetic loss generators for the 3 regimes
└── plots/                  # Generated evaluation figures (DPI 300)
    ├── regret_stochastic.png
    ├── regret_adversarial.png
    ├── regret_low_gap_stochastic.png
    ├── cum_loss_stochastic.png
    ├── cum_loss_adversarial.png
    └── cum_loss_low_gap_stochastic.png
```

---

## Reproducing the Results

### 1. Requirements
Ensure you have Python 3.9+ installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Experiments & Generate Plots
Execute `main.py` to run Monte Carlo trials across all regimes and update the figures in `plots/`:
```bash
python main.py
```

---

## Empirical Summary (Horizon $T = 5000$, $K = 10$)

| Regime | Metric | Minimax Hedge ($\eta^* = 0.061$) | AdaHedge (Adaptive) | Regret Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **Stochastic** | Final Regret $R_T$ | $39.86$ | **$5.96$** | **$6.7\times$ faster convergence ($\mathcal{O}(1)$)** |
| **Adversarial** | Final Regret $R_T$ | $34.91$ | **$7.48$** | **Smooth epoch tracking without runaway regret** |
| **Low-Gap** | Final Regret $R_T$ | $53.34$ | **$50.50$** | **Lowest regret without manual hyperparameter search** |

---

## Declaration of Authorship
*I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work. I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.*
