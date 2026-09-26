# Bayesian Optimization Research

Exploratory notebook comparing Bayesian optimization against traditional hyperparameter search strategies using [scikit-optimize](https://scikit-optimize.github.io/) (`skopt`).

## Contents

`research.ipynb` walks through:

1. **Bayesian optimization vs. random sampling** — minimizes a noisy synthetic "strength" function over a 2D search space with `gp_minimize`, plotting the best result found so far against random sampling as a baseline (see `bo_vs_random.png`).
2. **Core skopt API tour** — minimizers (`gp_minimize`, `forest_minimize`, `gbrt_minimize`, `dummy_minimize`), search space types (`Real`, `Integer`, `Categorical`), the `Optimizer` manual-loop interface, diagnostic plots (`plot_convergence`, `plot_evaluations`, `plot_objective`), and model persistence (`dump`/`load`).
3. **`BayesSearchCV`** — scikit-learn-compatible Bayesian hyperparameter search applied to:
   - `RandomForestClassifier` on the Wine dataset
   - `SVC` (polynomial kernel) on the Iris dataset
4. **Search strategy comparison** — `GridSearchCV`, `RandomizedSearchCV`, and `BayesSearchCV` benchmarked against each other on a `GradientBoostingClassifier` fit to the Breast Cancer dataset, comparing best cross-validated accuracy across an equivalent evaluation budget.

## Key takeaway

Bayesian optimization converges to a good result in fewer evaluations than random or grid search by using a surrogate model (Gaussian process, random forest, or gradient-boosted trees) to guide the next sample toward promising regions of the search space, rather than sampling blindly or exhaustively.

## Setup

```bash
pip install scikit-optimize scikit-learn numpy scipy matplotlib
```

Then open `research.ipynb` in Jupyter.

## Files

- `research.ipynb` — the notebook containing all experiments
- `bo_vs_random.png` — convergence plot comparing Bayesian optimization to random sampling
