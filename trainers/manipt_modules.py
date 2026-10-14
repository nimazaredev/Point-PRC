"""
ManiPT modules: Normalized Additive Fusion, ManiPT total loss,
and the Manifold Drift metric (Delta).

Notation follows the spec:
  z_3D, h_3D : (B, D) frozen / prompt-adapted 3D features (L2-normalized)
  z_txt, h_txt: (C, D) frozen / prompt-adapted text features (L2-normalized)
  w          : (C, D) LLM-enriched semantic class prototypes (L2-normalized)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NormalizedAdditiveFusion(nn.Module):
    """f = normalize(z + h), eps for numerical stability."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # z, h: (..., D)
        fused = z + h
        f = F.normalize(fused, p=2, dim=-1, eps=self.eps)  # (..., D)
        return f


class ManiPTLoss(nn.Module):
    """
    Implements:
        L_ce  = CrossEntropy(logits, targets)
        L_3D  = mean_i (1 - <f_3D_i, z_3D_i>)
        L_txt = mean_c (1 - <f_txt_c, w_c>)
        L_total = L_ce + lambda_reg * (L_3D + L_txt)
    """

    def __init__(self, lambda_reg: float = 1.0, tau: float = 100.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.tau = tau  # kept for API completeness; logits are usually
                         # already scaled by tau before being passed in

    def forward(self, f_3d, z_3d, f_txt, w_text_proto, logits, targets):
        """
        f_3d:  (B, D) fused 3D features
        z_3d:  (B, D) frozen 3D features
        f_txt: (C, D) fused text features
        w_text_proto: (C, D) LLM-enriched semantic prototypes
        logits: (B, C) tau * f_3d @ f_txt.T
        targets: (B,) int64 class indices
        """
        l_ce = F.cross_entropy(logits, targets)

        # L_3D: 1 - cosine sim per sample, averaged over batch
        cos_3d = torch.sum(f_3d * z_3d, dim=-1)          # (B,)
        l_3d = (1.0 - cos_3d).mean()

        # L_txt: 1 - cosine sim per class, averaged over classes
        cos_txt = torch.sum(f_txt * w_text_proto, dim=-1)  # (C,)
        l_txt = (1.0 - cos_txt).mean()

        total_loss = l_ce + self.lambda_reg * (l_3d + l_txt)
        return total_loss, l_ce, l_3d, l_txt



@torch.no_grad()
def calculate_manifold_drift(Z: torch.Tensor, H: torch.Tensor,
                              d_pca: int = 64, eps: float = 1e-12) -> float:
    """
    Z, H: (M, D) reference (frozen) / prompt-tuned features on the test set.
    Returns Delta = R(H) - R(Z)  (float).
    """
    assert Z.shape == H.shape, "Z and H must have the same shape (M, D)"
    Z = Z.double()
    H = H.double()

    mu = Z.mean(dim=0, keepdim=True)          # (1, D)
    Z_bar = Z - mu
    H_bar = H - mu

    # SVD on centered reference features
    # Z_bar = U diag(S) V^T ,  V: (D, D)
    U, S, Vh = torch.linalg.svd(Z_bar, full_matrices=False)
    V = Vh.T                                   # (D, r), r = min(M, D)
    d_pca = min(d_pca, V.shape[1])
    V_pca = V[:, :d_pca]                       # (D, d_pca)

    P_pca = V_pca @ V_pca.T                    # (D, D)
    I_D = torch.eye(P_pca.shape[0], dtype=P_pca.dtype, device=P_pca.device)
    off_manifold_proj = I_D - P_pca            # (D, D)

    def off_manifold_ratio(X_bar):
        off = X_bar @ off_manifold_proj.T       # (M, D)
        num = torch.sum(off ** 2)
        den = torch.sum(X_bar ** 2) + eps
        return (num / den).item()

    R_Z = off_manifold_ratio(Z_bar)
    R_H = off_manifold_ratio(H_bar)

    delta = R_H - R_Z
    return float(delta)








# ============================================================================
# [NEW - ADD]  EVERYTHING BELOW THIS LINE is new.
#              Paste from here to the end of the file at the END of your own
#              trainers/manipt_modules.py (and add `import math` at the top).
# ============================================================================
 
# ----------------------------------------------------------------------------
# [NEW - ADD] Rotation sampling
# ----------------------------------------------------------------------------
_AXIS_PAIR = {0: (1, 2), 1: (2, 0), 2: (0, 1)}   # cyclic pairs for x, y, z
 
 
def _rot_about_axis(angles: torch.Tensor, axis: int) -> torch.Tensor:
    """angles: (B,) in radians -> (B,3,3) right-handed rotation about a coordinate axis."""
    B = angles.shape[0]
    c, s = torch.cos(angles), torch.sin(angles)
    R = torch.eye(3, dtype=angles.dtype, device=angles.device).repeat(B, 1, 1)
    i, j = _AXIS_PAIR[axis]
    R[:, i, i] = c
    R[:, i, j] = -s
    R[:, j, i] = s
    R[:, j, j] = c
    return R
 
 
def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """q: (B,4) unit quaternion (w,x,y,z) -> (B,3,3)."""
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.view(-1, 3, 3)
 
 
def sample_rotations(B, mode, max_deg=180.0, axis="y", generator=None, device="cpu"):
    """
    mode:
      'none'       -> returns None (no rotation)
      'so2'        -> angle ~ U[-max_deg, max_deg] about `axis` (e.g. gravity axis)
      'so3'        -> uniform (Haar) rotation on SO(3); max_deg ignored
      'axis_angle' -> random axis, angle ~ U[-max_deg, max_deg]  (bounded SO(3) rotation)
    `generator` is a CPU torch.Generator so that the SAME rotations are produced
    in every experiment given the same seed (fair comparison E1/E2/E3/E4).
    """
    if mode == "none":
        return None
    if mode == "so2":
        a = (torch.rand(B, generator=generator) * 2 - 1) * math.radians(max_deg)
        R = _rot_about_axis(a, {"x": 0, "y": 1, "z": 2}[axis])
    elif mode == "so3":
        q = torch.randn(B, 4, generator=generator)
        q = F.normalize(q, dim=-1)
        R = _quat_to_matrix(q)
    elif mode == "axis_angle":
        ax = F.normalize(torch.randn(B, 3, generator=generator), dim=-1)
        th = (torch.rand(B, generator=generator) * 2 - 1) * math.radians(max_deg)
        zero = torch.zeros(B)
        K = torch.stack([zero, -ax[:, 2], ax[:, 1],
                         ax[:, 2], zero, -ax[:, 0],
                         -ax[:, 1], ax[:, 0], zero], dim=-1).view(B, 3, 3)
        I = torch.eye(3).expand(B, 3, 3)
        R = I + torch.sin(th)[:, None, None] * K + (1 - torch.cos(th))[:, None, None] * (K @ K)
    else:
        raise ValueError(f"Unknown rotation mode: {mode}")
    return R.to(device)
 
 
def apply_rotation(pts: torch.Tensor, R) -> torch.Tensor:
    """pts: (B,N,C>=3), R: (B,3,3) or None.  x' = x R^T (per sample)."""
    if R is None:
        return pts
    R = R.to(dtype=pts.dtype, device=pts.device)
    xyz = torch.bmm(pts[..., :3], R.transpose(1, 2))
    if pts.shape[-1] > 3:
        xyz = torch.cat([xyz, pts[..., 3:]], dim=-1)
    return xyz
 


# ----------------------------------------------------------------------------
# [NEW - ADD] Random isotropic scaling
# ----------------------------------------------------------------------------
def sample_scales(B, s_min=1.0, s_max=1.0, generator=None, device="cpu"):
    """s ~ U[s_min, s_max] per sample, returns (B,1,1) or None if scaling is off (s_min=s_max=1)."""
    if s_min == 1.0 and s_max == 1.0:
        return None
    s = s_min + (s_max - s_min) * torch.rand(B, generator=generator)
    return s.view(B, 1, 1).to(device)


def apply_scale(pts: torch.Tensor, s) -> torch.Tensor:
    """pts: (B,N,C>=3); scales xyz only (extra channels untouched)."""
    if s is None:
        return pts
    s = s.to(dtype=pts.dtype, device=pts.device)
    xyz = pts[..., :3] * s
    if pts.shape[-1] > 3:
        xyz = torch.cat([xyz, pts[..., 3:]], dim=-1)
    return xyz

# ----------------------------------------------------------------------------
# [NEW - ADD] Idea 1: penalty in rotation-sensitive directions
# ----------------------------------------------------------------------------
class RotationSubspace(nn.Module):
    """
    Computes the quadratic form  q(d) = d^T M d  for d = h - z  (B,D).
 
      mode='sens'       : M = P_rot = U_k U_k^T          (penalize drift in sensitive dirs)
      mode='complement' : M = I - P_rot                  (penalize drift only in insensitive dirs;
                                                           recommended partner of the invariance loss)
      mode='soft'       : M = D * Sigma_rot / tr(Sigma_rot) = U diag(D*lam/sum lam) U^T
 
    The file is produced offline by compute_rot_subspace.py and holds
    U (D,D) (columns sorted by decreasing eigenvalue) and evals (D,).
    """
 
    def __init__(self, path: str, mode: str = "sens", k: int = 0, energy: float = 0.9):
        super().__init__()
        assert mode in ("sens", "complement", "soft")
        blob = torch.load(path, map_location="cpu")
        U = blob["U"].float()
        ev = blob["evals"].float().clamp_min(0)
        D = U.shape[0]
        self.mode = mode
        if mode in ("sens", "complement"):
            if k <= 0:
                cum = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-12)
                k = int((cum < energy).sum().item()) + 1
            k = min(k, D)
            self.k = k
            self.register_buffer("Uk", U[:, :k].contiguous())
        else:
            self.k = D
            self.register_buffer("U", U.contiguous())
            self.register_buffer("w", (D * ev / ev.sum().clamp_min(1e-12)))
 
    def quad(self, d: torch.Tensor) -> torch.Tensor:
        if self.mode == "sens":
            return (d @ self.Uk).pow(2).sum(-1)
        if self.mode == "complement":
            return d.pow(2).sum(-1) - (d @ self.Uk).pow(2).sum(-1)
        return ((d @ self.U).pow(2) * self.w).sum(-1)
 
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """L = 1/2 * E[(h - z)^T M (h - z)]"""
        return 0.5 * self.quad(h - z).mean()
 
 
# ----------------------------------------------------------------------------
# [NEW - ADD] Idea 2: invariance between two rotated views
# ----------------------------------------------------------------------------
class RotationInvarianceLoss(nn.Module):
    """
    kind='h'  : 1 - <h1, h2>           (prompt-adapted features, default)
    kind='f'  : 1 - <f1, f2>           (fused features, which are what classify)
    kind='kl' : symmetric KL between softmax(logits_1) and softmax(logits_2)
    stopgrad  : SimSiam style, each side regresses to a detached copy of the other
    """
 
    def __init__(self, kind: str = "h", stopgrad: bool = False):
        super().__init__()
        assert kind in ("h", "f", "kl")
        self.kind = kind
        self.stopgrad = stopgrad
 
    @staticmethod
    def _cos(a, b):
        return (1.0 - (a * b).sum(-1)).mean()
 
    def forward(self, h1, h2, f1, f2, logit1, logit2):
        if self.kind in ("h", "f"):
            a, b = (h1, h2) if self.kind == "h" else (f1, f2)
            if self.stopgrad:
                return 0.5 * (self._cos(a, b.detach()) + self._cos(b, a.detach()))
            return self._cos(a, b)
 
        # symmetric KL
        l1, l2 = logit1, logit2
        if self.stopgrad:
            t1, t2 = l2.detach(), l1.detach()
        else:
            t1, t2 = l2, l1
        kl12 = F.kl_div(F.log_softmax(l1, -1), F.softmax(t1, -1), reduction="batchmean")
        kl21 = F.kl_div(F.log_softmax(l2, -1), F.softmax(t2, -1), reduction="batchmean")
        return 0.5 * (kl12 + kl21)

def _refill(xyz, keep_idx, gen):
    N, M = xyz.shape[0], keep_idx.shape[0]
    if M < N:
        extra = keep_idx[torch.randint(0, M, (N - M,), generator=gen)]
        keep_idx = torch.cat([keep_idx, extra])
    return xyz[keep_idx[:N]]


def corrupt_points(pts, gen, ops=("occ", "drop", "density", "outlier"),
                   max_drop=0.4, jitter_std=0.01):
    B, N, C = pts.shape
    out = pts.clone()
    for b in range(B):
        xyz = pts[b, :, :3]
        op = ops[int(torch.randint(0, len(ops), (1,), generator=gen))]
        u = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=gen))
        if op == "occ":        # occlusion محلی
            c = xyz[int(torch.randint(0, N, (1,), generator=gen))]
            k = int(N * u(0.1, max_drop))
            d = ((xyz - c) ** 2).sum(-1)
            keep = torch.topk(d, N - k, largest=True).indices
            new = _refill(xyz, keep, gen)
        elif op == "drop":     # نمونه‌گیری خلوت‌تر
            m = N - int(N * u(0.1, max_drop + 0.1))
            keep = torch.randperm(N, generator=gen)[:m].to(xyz.device)
            new = _refill(xyz, keep, gen)
        elif op == "density":  # تراکم غیریکنواخت
            c = xyz[int(torch.randint(0, N, (1,), generator=gen))]
            d = ((xyz - c) ** 2).sum(-1)
            w = torch.exp(-d / (d.mean() * u(0.5, 2.0) + 1e-8)).cpu() + 1e-4
            m = N - int(N * u(0.1, max_drop + 0.2))
            keep = torch.multinomial(w, m, replacement=False, generator=gen).to(xyz.device)
            new = _refill(xyz, keep, gen)
        else:                  # outlier
            n = int(N * u(0.02, 0.1))
            new = xyz.clone()
            idx = torch.randperm(N, generator=gen)[:n].to(xyz.device)
            new[idx] = (torch.rand(n, 3, generator=gen) * 2 - 1).to(xyz.device, xyz.dtype)
        if jitter_std > 0:
            new = new + (jitter_std * torch.randn(N, 3, generator=gen)).clamp(-0.05, 0.05).to(new.device, new.dtype)
        out[b, :, :3] = new
    return out