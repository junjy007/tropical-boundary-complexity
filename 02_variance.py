"""Theorem 2: Var(Z_L) from the recursion against Monte Carlo.

Run: python 02_variance.py [--quick]
"""
import argparse
import math

import torch

from core import (generator_energy, sample_network, sphere, var_Z_exact,
                  write_csv)

CASES = [
    (2, [128], [2.0, 2.0], 0.0),
    (4, [32] * 3, [2.0] * 4, 0.0),
    (4, [128] * 3, [2.0] * 4, 0.0),
    (4, [32] * 3, [2.0] * 4, 0.5),
    (8, [32] * 7, [2.0] * 8, 0.0),
    (8, [128] * 7, [2.0] * 8, 0.5),
    (5, [32, 128, 48, 96], [1.5, 2.5, 1.0, 3.0, 2.0], 0.5),
]


def var_z2_paper(p0, p1, s1, s2):
    """Depth-two closed form, equation (app_var_L2) of the paper."""
    pi = math.pi
    return (2.0 * s1 ** 2 * s2 ** 2 / (p0 * p1)
            * ((5.0 + 8.0 / pi) * (p0 + 2.0)
               - 2.0 * (1.0 + 1.0 / pi) ** 2 * p0))


def main(n_nets, p0=20, chunk=500):
    torch.manual_seed(1)
    rows = []
    print(f"{'L':>2} {'widths':>14} {'sb2':>4} {'Var exact':>11} "
          f"{'Var MC':>11} {'sem':>9} {'z':>6} {'CV':>7}")
    for L, widths, sig2, sb2 in CASES:
        values = []
        for start in range(0, n_nets, chunk):
            m = min(chunk, n_nets - start)
            layers = sample_network(p0, widths, sig2, sb2, m)
            values.append(generator_energy(layers, sphere(m, p0)))
        z = torch.cat(values)
        mean_exact, var_exact = var_Z_exact(p0, widths, sig2)
        if L == 2:
            closed = var_z2_paper(p0, widths[0], sig2[0], sig2[1])
            assert abs(closed - var_exact) < 1e-9 * closed
        var_mc = z.var().item()
        m4 = ((z - z.mean()) ** 4).mean().item()
        sem = math.sqrt(max(m4 - var_mc ** 2, 0.0) / n_nets)
        score = (var_mc - var_exact) / sem
        cv = math.sqrt(var_exact) / mean_exact
        print(f"{L:2d} {str(widths[:2]) + '..':>14} {sb2:4.1f} "
              f"{var_exact:11.4f} {var_mc:11.4f} {sem:9.4f} {score:6.2f} "
              f"{cv:7.3f}")
        rows.append(dict(L=L, widths=widths, sigma2=sig2, sigma_b2=sb2,
                         var_exact=var_exact, var_mc=var_mc, sem=sem, z=score,
                         cv_exact=cv, n_nets=n_nets))
    print("wrote", write_csv("02_variance.csv", rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(2000 if args.quick else 20000)
