"""Theorem 3(a, c): relative error delta = rho_exact / rho_L - 1.

rho_exact is estimated from E[r(S)] / pi over independent networks.
It is compared with the exact value at zero bias and with the
first-order value sum_l c_l / p_l for Gaussian biases, at p0 = 50,
R = 1, sigma_w^2 = 2, all hidden widths equal to p.

Run: python 04_finite_width_error.py [--quick]
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from core import (RESULTS, delta_first_order, delta_zero_bias, rho_L,
                  sample_r_of_S, write_csv)

P0, R, SW2 = 50, 1.0, 2.0
ZERO_BIAS_WIDTHS = (4, 6, 8, 10)
BIAS_WIDTHS = (8, 16, 32, 64, 128)
SB2 = 0.5


def estimate(n, L, p, sb2, chunk=50000):
    widths, sig2 = [p] * (L - 1), [SW2] * L
    r_target = math.pi * rho_L(sig2, sb2, P0, R)
    total, square, done = 0.0, 0.0, 0
    while done < n:
        m = min(chunk, n - done)
        r = sample_r_of_S(m, P0, widths, sig2, sb2, R) / r_target
        total += r.sum().item()
        square += (r * r).sum().item()
        done += m
    mean = total / n
    sem = math.sqrt(max(square / n - mean ** 2, 0.0) / n)
    return mean - 1.0, sem


def main(n_zero, n_bias):
    torch.manual_seed(3)
    rows = []
    print(f"{'L':>2} {'p':>4} {'sb2':>4} {'n':>9} {'delta MC':>11} "
          f"{'sem':>9} {'delta theory':>12} {'z':>6}")
    cases = [(L, p, 0.0, n_zero) for L in (4, 8) for p in ZERO_BIAS_WIDTHS]
    cases += [(L, p, SB2, n_bias) for L in (4, 8) for p in BIAS_WIDTHS]
    for L, p, sb2, n in cases:
        widths, sig2 = [p] * (L - 1), [SW2] * L
        delta, sem = estimate(n, L, p, sb2)
        if sb2 == 0.0:
            theory = delta_zero_bias(widths)
        else:
            theory = delta_first_order(widths, sig2, sb2, P0, R)
        score = (delta - theory) / sem
        print(f"{L:2d} {p:4d} {sb2:4.1f} {n:9d} {delta:11.4e} {sem:9.2e} "
              f"{theory:12.4e} {score:6.2f}", flush=True)
        rows.append(dict(L=L, p=p, sigma_b2=sb2, n_nets=n, delta_mc=delta,
                         sem=sem, delta_theory=theory, z=score))
    print("wrote", write_csv("04_finite_width_error.csv", rows))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for sb2, color in ((0.0, "k"), (SB2, "tab:green")):
        for L, marker in ((4, "o"), (8, "s")):
            sub = [r for r in rows if r["sigma_b2"] == sb2 and r["L"] == L]
            ax.errorbar([r["p"] for r in sub],
                        [abs(r["delta_mc"]) for r in sub],
                        yerr=[r["sem"] for r in sub], fmt=marker, color=color,
                        mfc="none", label=f"L={L}, $\\sigma_b^2$={sb2}")
            ax.plot([r["p"] for r in sub],
                    [abs(r["delta_theory"]) for r in sub], "-", color=color,
                    lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("hidden width p")
    ax.set_ylabel(r"$|\delta|$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "04_finite_width_error.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        main(100000, 20000)
    else:
        main(4000000, 400000)
