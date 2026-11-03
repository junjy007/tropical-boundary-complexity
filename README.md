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

## Results of the full run

The logs, tables and figures of one full run are in
`results/`.

Generator energy (20000 networks per case). All eight cases
agree with $\Gamma_L$, with $|z|\le 2.01$. They include bias
variances $0$ to $4$ and one case with unequal widths and
weight variances.

Variance (20000 networks per case). All seven cases agree with
the recursion, with $|z|\le 2.13$.

Crossing density ($L=8$, $p_0=50$, width $600$, $200$ networks
with $25$ arcs of $600$ samples each). On $\sigma_b^2=0.5$, the
counted density over $\rho_L$ is between $0.90$ and $1.43$ for
$\sigma_w^2$ from $1$ to $4.5$. At $\sigma_w^2=0.5$, where
$\rho_L=0.0003$, the run counted no crossing. At zero bias
and $\sigma_w^2=2$, the ratio is $1.11$ ($z=1.66$). Crossings
are rare and vary strongly between networks, so these ratios
carry large standard errors, listed in the table.

Relative error $\delta$, zero bias ($4\times 10^6$ networks per
point). The estimates follow the exact value, with $|z|\le 0.89$
except $L=8$, $p=4$, where $z=-3.6$. The variable $r(\mathbf S)$
is heavy-tailed at small widths, so the standard error is itself
uncertain there.

Relative error $\delta$, $\sigma_b^2=0.5$ ($4\times 10^5$ networks
per point). The value $\sum_l c_l/p$ is first order in $1/p$. At
$L=4$ the estimates agree with it at $p\ge 16$. At $L=8$ they
differ from it by about $5.5/p^2$ at $p\le 64$ and agree at
$p=128$. A difference of this size is the second-order term that
the first-order value leaves out.

## License

MIT, see `LICENSE`.
