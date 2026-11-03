"""Theorem 3(b): counted crossings against the estimate rho_L.

Counts sign changes of g along great-circle arcs, at L = 8, p0 = 50,
R = 1, hidden width 600, on the slice sigma_b^2 = 0.5 of the phase
diagram, and also at zero bias, where rho_L = 1 / (pi R). Draws the
heat map of rho_L with the counted values on the slice.

Run: python 03_crossing_density.py [--quick]
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import RESULTS, count_crossings, rho_L, sample_network, write_csv

P0, R = 50, 1.0
SWEEP = [(s, 0.5) for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)]
SWEEP += [(2.0, 0.0)]


def measure(L, width, sw2, sb2, n_nets, n_arcs, n_samples):
    widths, sig2 = [width] * (L - 1), [sw2] * L
    values = []
    for _ in range(n_nets):
        layers = sample_network(P0, widths, sig2, sb2, 1, dtype=torch.float32)
        values.append(count_crossings(layers, P0, R, n_arcs, n_samples))
    values = torch.cat(values).double()
    return values.mean().item(), values.std().item() / math.sqrt(n_nets)


def figure(L, rows):
    sw = np.linspace(0.5, 4.5, 81)
    sb = np.linspace(0.0, 2.0, 81)
    grid = np.array([[rho_L([a] * L, b, P0, R) for a in sw] for b in sb])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    mesh = ax.pcolormesh(sw, sb, grid, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=r"$\rho_L$")
    ax.axvline(2.0, color="w", ls="--", lw=1)
    ax.set_xlabel(r"$\sigma_w^2$")
    ax.set_ylabel(r"$\sigma_b^2$")
    ax.set_title(f"L = {L}: numbers are counted density / $\\rho_L$")
    for row in rows:
        ax.annotate(f"{row['counted'] / row['rho_L']:.2f}",
                    (row["sigma_w2"], row["sigma_b2"]),
                    color="w", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "03_phase_diagram.pdf")


def main(n_nets, n_arcs=25, n_samples=600, L=8, width=600):
    torch.manual_seed(2)
    rows = []
    print(f"{'sw2':>4} {'sb2':>4} {'rho_L':>8} {'counted':>8} {'sem':>8} "
          f"{'ratio':>6}")
    for sw2, sb2 in SWEEP:
        predicted = rho_L([sw2] * L, sb2, P0, R)
        mean, sem = measure(L, width, sw2, sb2, n_nets, n_arcs, n_samples)
        print(f"{sw2:4.1f} {sb2:4.1f} {predicted:8.4f} {mean:8.4f} "
              f"{sem:8.4f} {mean / predicted:6.3f}", flush=True)
        rows.append(dict(L=L, width=width, sigma_w2=sw2, sigma_b2=sb2,
                         rho_L=predicted, counted=mean, sem=sem,
                         n_nets=n_nets, n_arcs=n_arcs, n_samples=n_samples))
    print("wrote", write_csv("03_crossing_density.csv", rows))
    figure(L, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(10 if args.quick else 200)
