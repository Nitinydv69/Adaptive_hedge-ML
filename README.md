# Online Learning with Adaptive Hedge

**Course:** Statistical Methods for Machine Learning (A.Y. 2025/26)
**Instructor:** Prof. Nicolò Cesa-Bianchi
**Author:** Nitin Yadav (University of Milan)
**Assignment:** Experimental Project 1, Online Learning (Adaptive Hedge)

---

## Overview

This repository compares **Hedge** with a fixed learning rate and **AdaHedge**
(van Erven et al., NeurIPS 2011; de Rooij et al., JMLR 2014) in prediction with
expert advice. Both algorithms are implemented from scratch with NumPy.

Hedge needs a learning rate `eta` chosen in advance. The theory-optimal value
depends on the horizon `T` and is tuned for the worst case, so it is slow on easy
data. AdaHedge sets `eta_t = ln(K) / Delta_{t-1}` online, where `Delta` is the
cumulative mixability gap, so it needs no tuning.

## Loss sequences (K = 10 experts, T = 5000 rounds)

1. **Stochastic:** one expert Bernoulli(0.3), nine Bernoulli(0.5). Large gap.
2. **Low gap:** one expert Bernoulli(0.49), nine Bernoulli(0.50). Gap of 0.01.
3. **Adversarial (switching):** three segments, and the best expert changes in each.
   The sequence is random and oblivious.
4. **Alternating (extra):** deterministic. The leader flips every round, which is
   a worst case for follow-the-leader and for large `eta`.

## Repository structure

```
.
├── README.md
├── report.tex / report.pdf   # Final report
├── requirements.txt          # numpy, matplotlib
├── main.py                   # Runs all experiments, saves plots and results/summary.csv
├── check_bound.py            # Checks R_T <= 2*Delta_T for AdaHedge (report, Table 3)
├── src/
│   ├── algorithms.py         # Hedge and AdaHedge
│   └── datasets.py           # The four loss generators
├── plots/                    # regret_*, cum_loss_*, eta_sweep_*, adahedge_eta.png
└── results/
    └── summary.csv           # Final regret (mean, std) per setting and algorithm
```

## Reproducing the results

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python main.py          # figures in plots/, table in results/summary.csv
python check_bound.py   # regret vs. mixability-gap check
```

All runs use seeds 0-19, and every algorithm sees the same sequence for a given seed.
The runtime is a few minutes. To build the report: `latexmk -pdf report.tex`.

## Results (final regret at T = 5000, mean ± std over 20 seeds)

| Setting     | Hedge η=0.005 | Hedge η\*=0.061 | Hedge η=0.5 |   **AdaHedge** |
| :---------- | ------------: | --------------: | ----------: | -------------: |
| Stochastic  |   450.2 ± 1.8 |      39.9 ± 0.2 |   6.9 ± 0.6 |  **5.5 ± 1.8** |
| Low gap     |   61.7 ± 15.0 |      54.6 ± 5.4 | 53.6 ± 18.8 |    54.3 ± 11.5 |
| Switching   |  229.1 ± 11.9 |      37.7 ± 3.4 |  12.1 ± 6.2 |  **9.6 ± 6.0** |
| Alternating |   325.3 ± 0.0 |      64.7 ± 0.0 | 309.6 ± 0.0 | **54.1 ± 0.0** |

Here `eta* = sqrt(8 ln K / T)` is the worst-case-optimal fixed rate.

## Main findings

- **No single fixed `eta` is good everywhere.** A large `eta` is excellent on the
  stochastic data but has regret around 310 on the alternating sequence. The safe
  `eta*` is about 7x worse than AdaHedge on the stochastic data.
- **AdaHedge is close to the best fixed choice in every setting**, without knowing `T`
  or how hard the data is.
- **Low gap:** no algorithm can be distinguished from another. The differences are
  smaller than the variation across seeds, because the regret is dominated by the
  lucky best expert, not by the learner.
- **Learning-rate behaviour:** on easy data `eta_t` stays large because the mixability
  gap stops growing. On hard data it decays roughly like `1/sqrt(t)`.
- **Theory check:** `R_T <= 2*Delta_T` held in all 20 runs of every setting.

## Limitations

Only K = 10, T = 5000 and 20 seeds. The `eta` grid is coarse, the adversarial sequence
is oblivious, and there are no significance tests. See Section 5 of the report.

## Declaration of Authorship

_I declare that this material, which I now submit for assessment, is entirely my own
work and has not been taken from the work of others, save and to the extent that such
work has been cited and acknowledged within the text of my work. I understand that
plagiarism, collusion, and copying are grave and serious offences in the university and
accept the penalties that would be imposed should I engage in plagiarism, collusion or
copying. This assignment, or any part of it, has not been previously submitted by me or
any other person for assessment on this or any other course of study._
