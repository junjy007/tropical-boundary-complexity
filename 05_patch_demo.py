"""Level curves of g on a 2D patch of the sphere S^49.

Rows vary the depth L, columns the weight variance sigma_w^2, with
width 600 and sigma_b^2 = 0.5. One random network per panel. The
field shown is g / sqrt(2 q^(L)), with one level spacing everywhere.

Run: python 05_patch_demo.py [--quick]
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import RESULTS, logit_gap, q_L, sample_network

P0, R, SB2, WIDTH, DELTA = 50, 3.0, 0.5, 600, 0.04
DEPTHS = (2, 3, 4, 8, 16)
VARIANCES = (0.5, 1.0, 2.0, 4.0, 8.0)


def main(n_grid):
    torch.manual_seed(4)
    q, _ = torch.linalg.qr(torch.randn(P0, 3, dtype=torch.float32))
    a = np.linspace(-0.6, 0.6, n_grid)
    A, B = np.meshgrid(a, a)
    A_t = torch.tensor(A, dtype=torch.float32).reshape(-1, 1)
    B_t = torch.tensor(B, dtype=torch.float32).reshape(-1, 1)
    pts = q[:, 0] + A_t * q[:, 1] + B_t * q[:, 2]
    pts = R * pts / pts.norm(dim=-1, keepdim=True)
    fig, axes = plt.subplots(len(DEPTHS), len(VARIANCES), figsize=(9, 9))
    for i, L in enumerate(DEPTHS):
        for j, sw2 in enumerate(VARIANCES):
            sig2 = [sw2] * L
            layers = sample_network(P0, [WIDTH] * (L - 1), sig2, SB2, 1,
                                    dtype=torch.float32)
            g = logit_gap(layers, pts.unsqueeze(0)).squeeze(0)
            field = g.numpy().reshape(A.shape)
            field = field / math.sqrt(2.0 * q_L(sig2, SB2, P0, R))
            low = np.floor(field.min() / DELTA) * DELTA
            levels = np.arange(low, field.max() + DELTA, DELTA)
            ax = axes[i, j]
            vmax = max(abs(field.min()), abs(field.max()))
            ax.imshow(field, extent=(-0.6, 0.6, -0.6, 0.6), origin="lower",
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            if len(levels) > 1:
                ax.contour(A, B, field, levels=levels[:400], colors="k",
                           linewidths=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(rf"$\sigma_w^2$={sw2}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"L={L}", fontsize=9)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "05_patch_demo.png"
    fig.savefig(path, dpi=200)
    print("wrote", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(100 if args.quick else 300)
