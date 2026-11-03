# Tropical Boundary Complexity of Deep ReLU Binary Classifiers

Numerical checks for the NeurIPS 2026 paper of the same title
by Jun Li. The package is small and covers the main results.
It samples random fully connected ReLU networks with two
linear outputs at initialization and compares the theorems
with Monte Carlo estimates.

| Script | Paper item | What it does |
|---|---|---|
| `01_generator_energy.py` | Theorem 1 | $E[Z_L]$ against $\Gamma_L$, over depth, width, weight variance and bias variance |
| `02_variance.py` | Theorem 2 | $\mathrm{Var}(Z_L)$ from the recursion against Monte Carlo |
| `03_crossing_density.py` | Theorem 3(b), phase diagram figure | Counted crossings along great-circle arcs against $\rho_L$ |
| `04_finite_width_error.py` | Theorem 3(a, c) | $\delta=\rho_{\mathrm{exact}}/\rho_L-1$ against its exact value (zero bias) and first-order value (Gaussian biases) |
| `05_patch_demo.py` | Level-curve figure | Level curves of $g$ on a patch of the sphere |

`core.py` holds the closed forms, the network sampler and the
estimators.

## Usage

```
pip install -r requirements.txt
python 01_generator_energy.py
python 02_variance.py
python 03_crossing_density.py
python 04_finite_width_error.py
python 05_patch_demo.py
```

Every script accepts `--quick` for a smoke test of seconds.
The full runs take minutes on a laptop CPU. Each script fixes
its random seed and writes to `results/`.

## Scope

These scripts are a compact version of the experiments behind
the paper. They use fewer networks than some figures in the
paper, whose zero-bias $\delta$ curves use up to $5\times 10^8$
networks per point, and the plots are not styled like the
published figures.

## License

MIT, see `LICENSE`.
