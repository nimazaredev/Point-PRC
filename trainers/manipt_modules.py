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