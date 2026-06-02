"""Per-patch classical nonlinear feature maps (poly-2, RFF).

These are the classical-nonlinear controls for the question "is the
low-capacity quantum advantage actually quantum, or just nonlinearity?".
Each map unfolds 3x3 stride-3 patches (the same patches the quantum kernel and
the random-conv baseline see), applies a fixed nonlinear map per patch, and
returns a (N, F, H_out, W_out) spatial tensor with the same layout as the
quantum feature cache, so the existing heads and the LinearSVC probe consume it
unchanged.

- poly2: 9 single pixels x_i + 36 pairwise products x_i x_j (i<j) = 45 features,
  the exact structural and dimensional match to quantum Z + ZZ at 1-kernel scale
  (squares x_i^2 are omitted: their quantum analogue <Z_i Z_i> = 1 is constant).
- rff: random Fourier features sqrt(2/D) cos(Wx+b) approximating an RBF kernel,
  at tunable dimension D (covers both the 45 and 180 matched scales).

Inputs are the raw [0,1] images from the dataloader (ToTensor only), identical
to what the random-conv baseline receives.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _unfold(images, k=3, s=3):
    """(N,1,H,W) -> patches (N, P, k*k) and output grid (h_out, w_out)."""
    x = images.float()
    h_out = (x.shape[2] - k) // s + 1
    w_out = (x.shape[3] - k) // s + 1
    patches = F.unfold(x, kernel_size=k, stride=s)  # (N, k*k, P)
    return patches.transpose(1, 2), h_out, w_out    # (N, P, k*k)


def poly2_features(images, k=3, s=3):
    """Per-patch degree-2 monomials (linear + pairwise), 45-dim for k=3."""
    patches, h, w = _unfold(images, k, s)           # (N, P, d)
    n, p, d = patches.shape
    iu = torch.triu_indices(d, d, offset=1)          # C(d,2) pairs
    cross = patches[:, :, iu[0]] * patches[:, :, iu[1]]
    feat = torch.cat([patches, cross], dim=2)        # (N, P, d + C(d,2))
    return feat.transpose(1, 2).reshape(n, feat.shape[2], h, w)


def rff_features(images, n_output, gamma=None, seed=42, k=3, s=3):
    """Per-patch random Fourier features approximating an RBF kernel."""
    patches, h, w = _unfold(images, k, s)            # (N, P, d)
    n, p, d = patches.shape
    if gamma is None:
        gamma = 1.0 / d                              # standard heuristic
    rng = np.random.default_rng(seed)
    W = torch.from_numpy((rng.standard_normal((d, n_output))
                          * np.sqrt(2 * gamma)).astype(np.float32))
    b = torch.from_numpy(rng.uniform(0, 2 * np.pi, n_output).astype(np.float32))
    proj = patches @ W + b                           # (N, P, D)
    z = np.sqrt(2.0 / n_output) * torch.cos(proj)
    return z.transpose(1, 2).reshape(n, n_output, h, w)
