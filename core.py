"""Shared functions for the numerical checks.

Notation follows the paper. A depth-L network has L-1 ReLU hidden
layers and a linear output layer with two units, g = f_1 - f_2,
W^(l)_ij ~ N(0, sigma_l^2 / p_{l-1}), b^(l)_i ~ N(0, sigma_{b,l}^2).
"""
import csv
import math
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent / "results"
DTYPE = torch.float64


def write_csv(name, rows):
    """Write a list of dicts to results/<name> and return the path."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def as_lists(L, width, sigma2):
    """Expand scalars to per-layer lists: hidden widths and variances."""
    widths = [width] * (L - 1) if isinstance(width, int) else list(width)
    sig2 = [sigma2] * L if isinstance(sigma2, (int, float)) else list(sigma2)
    assert len(widths) == L - 1 and len(sig2) == L
    return widths, sig2


# ---------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------
def gamma_L(sig2):
    """Theorem 1: Gamma_L = 8 (1 + 1/pi) prod_l sigma_l^2 / 2."""
    value = 8.0 * (1.0 + 1.0 / math.pi)
    for s in sig2:
        value *= s / 2.0
    return value


def q_L(sig2, sb2, p0, R):
    """Pre-activation variance q^(L) with common bias variance sb2."""
    q = sig2[0] * R ** 2 / p0 + sb2
    for s in sig2[1:]:
        q = s / 2.0 * q + sb2
    return q


def rho_L(sig2, sb2, p0, R):
    """Theorem 3(b): the estimate of the crossing density."""
    kappa = 1.0 / (2.0 * math.pi * math.sqrt((1.0 + 1.0 / math.pi) * p0))
    return kappa * math.sqrt(gamma_L(sig2) / q_L(sig2, sb2, p0, R))


def var_Z_exact(p0, widths, sig2):
    """Theorem 2: Var(Z_L) at finite width, by the recursion on (P, Q).

    The bottom-up state is A1 = E[tr N], P = E[(tr N)^2],
    Q = E[tr N^2], N = Phi^T Phi. It is equivalent to the recursion
    stated in the paper and uses the same linear map on (P, Q).
    Returns (Gamma_L, Var(Z_L)).
    """
    L = len(sig2)
    A1, P, Q = float(p0), float(p0) ** 2, float(p0)
    fan_in = p0
    for l in range(L - 2):
        n, tau = widths[l], sig2[l] / fan_in
        A1 = n * tau / 2.0 * A1
        P, Q = (tau ** 2 * ((n * n + n) / 4.0 * P + n * Q),
                tau ** 2 * (n / 2.0 * P + (n * n + 3.0 * n) / 4.0 * Q))
        fan_in = n
    n = widths[L - 2]
    alpha = sig2[L - 2] / fan_in
    s2 = sig2[L - 1] / n
    e_t = 2.0 * s2 * (1.0 + 1.0 / math.pi)
    e_t2 = s2 ** 2 * (10.0 + 16.0 / math.pi)
    mean = n * e_t * alpha * A1
    second = alpha ** 2 * (n * e_t2 * (P + 2.0 * Q)
                           + n * (n - 1.0) * e_t ** 2 * P)
    return mean, second - mean ** 2


def delta_zero_bias(widths):
    """Theorem 3(c), biases zero: exact relative error of rho_L."""
    value = 1.0
    for p in widths:
        value *= 1.0 - (1.0 + p) / 2.0 ** p
    return value - 1.0


def delta_first_order(widths, sig2, sb2, p0, R):
    """Theorem 3(c), Gaussian biases: sum_l c_l / p_l, first order.

    c_l = 5/8 (3 omega_l + 1)(omega_l - 1),
    omega_l = q^(l) / q^(L) * prod_{m > l} sigma_m^2 / 2.
    """
    L = len(sig2)
    q = [sig2[0] * R ** 2 / p0 + sb2]
    for s in sig2[1:]:
        q.append(s / 2.0 * q[-1] + sb2)
    total = 0.0
    for l in range(L - 1):
        omega = q[l] / q[L - 1]
        for s in sig2[l + 1:]:
            omega *= s / 2.0
        total += 0.625 * (3.0 * omega + 1.0) * (omega - 1.0) / widths[l]
    return total


# ---------------------------------------------------------------------
# Random networks
# ---------------------------------------------------------------------
def sample_network(p0, widths, sig2, sb2, n_nets=1, dtype=DTYPE):
    """Batch of n_nets networks; list of (W, b) with batch dimension."""
    dims = [p0] + list(widths) + [2]
    layers = []
    for l in range(len(dims) - 1):
        W = torch.randn(n_nets, dims[l + 1], dims[l], dtype=dtype)
        W *= math.sqrt(sig2[l] / dims[l])
        b = torch.randn(n_nets, dims[l + 1], dtype=dtype) * math.sqrt(sb2)
        layers.append((W, b))
    return layers


def logit_gap(layers, x):
    """g(x) for x of shape (n_nets, n_points, p0)."""
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W.transpose(1, 2) + b.unsqueeze(1)
        if i < len(layers) - 1:
            h = torch.relu(h)
    return h[..., 0] - h[..., 1]


def generator_energy(layers, x):
    """Z_L = ||G_1||_F^2 + ||G_2||_F^2 on the cell of x, one x per net.

    x has shape (n_nets, p0). The matrix A Phi is built bottom-up with
    the activation indicators of layers 1, ..., L-2.
    """
    h = x
    A = layers[0][0]
    for l in range(len(layers) - 2):
        W, b = layers[l]
        z = (h.unsqueeze(1) @ W.transpose(1, 2)).squeeze(1) + b
        active = (z > 0).to(h.dtype)
        h = torch.relu(z)
        A = layers[l + 1][0] @ (active.unsqueeze(-1) * A)
    B = layers[-1][0]
    bp, bn = B.clamp(min=0), (-B).clamp(min=0)
    beta1 = bp[:, 0] + bn[:, 1]
    beta2 = bp[:, 1] + bn[:, 0]
    return ((beta1 ** 2 + beta2 ** 2) * (A ** 2).sum(-1)).sum(-1)


def sphere(n, p0, R=1.0, dtype=DTYPE):
    """n uniform points on the sphere of radius R in R^p0."""
    x = torch.randn(n, p0, dtype=dtype)
    return R * x / x.norm(dim=-1, keepdim=True)


def count_crossings(layers, p0, R, n_arcs, n_samples):
    """Sign changes of g along random great-circle arcs.

    Returns crossings per unit length for each network, as total
    crossings over total arc length.
    """
    n_nets = layers[0][0].shape[0]
    dtype = layers[0][0].dtype
    u = sphere(n_nets * n_arcs, p0, 1.0, dtype).view(n_nets, n_arcs, p0)
    v = sphere(n_nets * n_arcs, p0, 1.0, dtype).view(n_nets, n_arcs, p0)
    cos = (u * v).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    tangent = v - cos * u
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)
    t = torch.linspace(0.0, 1.0, n_samples, dtype=u.dtype).view(1, 1, -1, 1)
    s = t * theta.unsqueeze(2)
    pts = R * (torch.cos(s) * u.unsqueeze(2) + torch.sin(s) * tangent.unsqueeze(2))
    g = logit_gap(layers, pts.view(n_nets, n_arcs * n_samples, p0))
    g = g.view(n_nets, n_arcs, n_samples)
    sign = torch.sign(g)
    changes = (sign[..., 1:] != sign[..., :-1]).sum(dim=(1, 2)).to(u.dtype)
    return changes / (R * theta.squeeze(-1).sum(-1))


def sample_r_of_S(n, p0, widths, sig2, sb2, R):
    """Draw r(S) = sqrt(det S) / S_11 for n independent networks.

    Theorem 3(a) gives rho_exact = E[r(S)] / pi. The rows of W^(l) are
    independent, so the pair (z_i, z_i') of value and arc derivative
    of unit i is Gaussian given the layer below, and only the Gram
    matrix (u, v, w) of (h, h') has to be propagated. This uses no
    approximation and costs far less than building the weights.
    """
    u = torch.full((n,), float(R ** 2), dtype=DTYPE)
    v = torch.zeros(n, dtype=DTYPE)
    w = torch.ones(n, dtype=DTYPE)
    fan_in = p0
    for l, p in enumerate(widths):
        tau = sig2[l] / fan_in
        a = tau * u + sb2                       # Var(z)
        c = tau * v                             # Cov(z, z')
        d = tau * w                             # Var(z')
        sa = torch.sqrt(a)
        safe = torch.where(sa > 0, sa, torch.ones_like(sa))
        beta = c / safe
        rest = torch.sqrt((d - beta ** 2).clamp(min=0.0))
        e1 = torch.randn(n, p, dtype=DTYPE)
        e2 = torch.randn(n, p, dtype=DTYPE)
        z = sa.unsqueeze(1) * e1
        dz = beta.unsqueeze(1) * e1 + rest.unsqueeze(1) * e2
        active = (z > 0).to(DTYPE)
        h, dh = z * active, dz * active
        u = (h * h).sum(1)
        v = (h * dh).sum(1)
        w = (dh * dh).sum(1)
        fan_in = p
    scale = 2.0 * sig2[-1] / fan_in
    s11 = scale * u + 2.0 * sb2
    s12 = scale * v
    s22 = scale * w
    det = (s11 * s22 - s12 ** 2).clamp(min=0.0)
    safe = torch.where(s11 > 0, s11, torch.ones_like(s11))
    return torch.where(s11 > 0, torch.sqrt(det) / safe, torch.zeros_like(s11))
