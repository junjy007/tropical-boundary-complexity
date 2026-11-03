"""Theorem 1: E[Z_L] = Gamma_L, for any width and any bias variance.

Run: python 01_generator_energy.py [--quick]
"""
import argparse
import math

import torch

from core import gamma_L, generator_energy, sample_network, sphere, write_csv

CASES = [
    # L, hidden widths, weight variances, bias variance
    (2, [128], [2.0, 2.0], 0.0),
    (4, [64] * 3, [2.0] * 4, 0.0),
    (4, [64] * 3, [2.0] * 4, 0.5),
    (4, [64] * 3, [2.0] * 4, 4.0),
    (8, [64] * 7, [2.0] * 8, 0.5),
    (4, [64] * 3, [1.0] * 4, 0.5),
    (4, [64] * 3, [3.0] * 4, 0.5),
    (5, [32, 128, 48, 96], [1.5, 2.5, 1.0, 3.0, 2.0], 0.5),
]


def main(n_nets, p0=20, chunk=1000):
    torch.manual_seed(0)
    rows = []
    print(f"{'L':>2} {'widths':>18} {'sb2':>4} {'Gamma_L':>10} "
          f"{'MC mean':>10} {'sem':>9} {'z':>6}")
    for L, widths, sig2, sb2 in CASES:
        values = []
        for start in range(0, n_nets, chunk):
            m = min(chunk, n_nets - start)
            layers = sample_network(p0, widths, sig2, sb2, m)
            values.append(generator_energy(layers, sphere(m, p0)))
        z = torch.cat(values)
        mean, sem = z.mean().item(), z.std().item() / math.sqrt(n_nets)
        gamma = gamma_L(sig2)
        score = (mean - gamma) / sem
        print(f"{L:2d} {str(widths[:2]) + '..':>18} {sb2:4.1f} {gamma:10.4f} "
              f"{mean:10.4f} {sem:9.4f} {score:6.2f}")
        rows.append(dict(L=L, widths=widths, sigma2=sig2, sigma_b2=sb2,
                         gamma_L=gamma, mc_mean=mean, sem=sem, z=score,
                         n_nets=n_nets))
    print("wrote", write_csv("01_generator_energy.csv", rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(2000 if args.quick else 20000)
