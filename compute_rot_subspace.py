"""
[NEW - ADD] This is a brand-new FILE. Save it as compute_rot_subspace.py in the project root
(same level as train.py). It does not modify anything existing.

Offline, training-free computation of the rotation-sensitive subspace of the FROZEN ULIP.

  z(x)      = normalize(E_0(x) P)
  zbar(x)   = E_R z(Rx)
  Sigma_rot = E_x E_R [ (z(Rx) - zbar(x)) (z(Rx) - zbar(x))^T ]
  Sigma_B   = Cov_c( mu_c ),  mu_c = mean of zbar(x) over class c      (only for --semantic)

Saves {'U': (D,D), 'evals': (D,)} sorted by decreasing eigenvalue and prints a
spectrum diagnostic (if the spectrum is flat, idea 1 degenerates to M ~ I and is useless).

Usage (same dataset/config args as train.py, e.g. few-shot train split of the source set):
  python compute_rot_subspace.py \
      --root data/fewshot/shapenetcorev2 \
      --dataset-config-file configs/datasets/shapenetcorev2.yaml \
      --config-file configs/trainers/PointPRC_ManiPT/custom_ulip.yaml \
      --rot-mode so3 --n-rot 16 --out output/rot_subspace/snv2_so3.pt \
      DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 32

Use ONLY the train split (never test) to avoid leakage.
"""
import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import train as T                      # registers datasets/trainers, gives setup_cfg
from dassl.data import DataManager
from trainers.pointprc import load_ulip_to_cpu
from trainers.manipt_modules import sample_rotations, apply_rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset-config-file", default="")
    p.add_argument("--config-file", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ulip-version", default="ulip2", choices=["ulip1", "ulip2"])
    p.add_argument("--rot-mode", default="so3", choices=["so2", "so3", "axis_angle"])
    p.add_argument("--rot-max-deg", type=float, default=180.0)
    p.add_argument("--rot-axis", default="z")
    p.add_argument("--n-rot", type=int, default=16, help="rotations per sample")
    p.add_argument("--max-batches", type=int, default=0, help="0 = whole train loader")
    p.add_argument("--semantic", action="store_true",
                   help="generalized eigenproblem Sigma_rot u = lam (Sigma_B + eps I) u")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--out", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()

    args = SimpleNamespace(
        dataset_config_file=a.dataset_config_file, config_file=a.config_file, root=a.root,
        output_dir="", resume="", seed=a.seed, source_domains=None, target_domains=None,
        transforms=None, trainer="", backbone="", head="", max_epoch=0, eval_only=False,
        no_mac=False, no_tdc=False, no_mec=False, ulip_version=a.ulip_version, opts=a.opts)
    cfg = T.setup_cfg(args)

    dm = DataManager(cfg)
    frozen = load_ulip_to_cpu(cfg, zero_shot_model=True).cuda().eval()
    P = frozen.pc_projection
    gen = torch.Generator().manual_seed(a.seed + 777)

    @torch.no_grad()
    def feat(x):
        return F.normalize(frozen.point_encoder(x) @ P, dim=-1)

    D = P.shape[1]
    S = torch.zeros(D, D, dtype=torch.float64, device="cuda")
    n = 0
    cls_sum, cls_cnt = {}, {}

    for bi, batch in enumerate(dm.train_loader_x):
        if a.max_batches and bi >= a.max_batches:
            break
        x = batch["pc"].cuda()
        y = batch["label"]
        zs = []
        for _ in range(a.n_rot):
            R = sample_rotations(x.shape[0], a.rot_mode, a.rot_max_deg, a.rot_axis, gen, x.device)
            zs.append(feat(apply_rotation(x, R)))
        Z = torch.stack(zs, 1)                       # (B, n_rot, D)
        zbar = Z.mean(1, keepdim=True)
        dZ = (Z - zbar).reshape(-1, D).double()
        S += dZ.T @ dZ
        n += dZ.shape[0]
        zb = zbar.squeeze(1).double()
        for c in y.unique().tolist():
            m = (y == c).cuda()
            cls_sum[c] = cls_sum.get(c, 0) + zb[m].sum(0)
            cls_cnt[c] = cls_cnt.get(c, 0) + int(m.sum())
        if bi % 20 == 0:
            print(f"batch {bi}", flush=True)

    S = (S / n).cpu()

    if a.semantic:
        import scipy.linalg as sla
        mus = torch.stack([cls_sum[c] / cls_cnt[c] for c in sorted(cls_sum)]).cpu()
        mu = mus.mean(0, keepdim=True)
        SB = ((mus - mu).T @ (mus - mu)) / mus.shape[0]
        w, V = sla.eigh(S.numpy(), (SB + a.eps * torch.eye(D, dtype=SB.dtype)).numpy())
        order = w.argsort()[::-1]
        w, V = w[order].copy(), torch.from_numpy(V[:, order].copy())
        U, _ = torch.linalg.qr(V)                    # orthonormal basis, leading span preserved
        evals = torch.from_numpy(w).clamp_min(0)
        # NOTE: with --semantic, 'soft' weights are approximate (generalized eigenvalues
        # attached to an orthonormalized basis); prefer 'sens'/'complement' modes.
    else:
        evals, U = torch.linalg.eigh(S)
        evals, U = evals.flip(0), U.flip(1)
        evals = evals.clamp_min(0)

    # ---------------- diagnostic ----------------
    cum = torch.cumsum(evals, 0) / evals.sum()
    pr = (evals.sum() ** 2 / (evals ** 2).sum()).item()                  # participation ratio
    ent = -(evals / evals.sum() * torch.log(evals / evals.sum() + 1e-12)).sum()
    print("\n=== rotation-sensitivity spectrum ===")
    print(f"tr(Sigma_rot)={S.trace():.4f}   D={D}")
    for k in (1, 4, 8, 16, 32, 64, 128):
        if k <= D:
            print(f"top-{k:<3d} energy: {cum[k - 1]:.3f}")
    for thr in (0.5, 0.8, 0.9, 0.95):
        print(f"k for {int(thr * 100)}% energy: {int((cum < thr).sum()) + 1}")
    print(f"participation ratio: {pr:.1f}   effective rank (exp H): {torch.exp(ent):.1f}   (flat => ~D={D})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"U": U.float(), "evals": evals.float(),
                "meta": {"rot_mode": a.rot_mode, "n_rot": a.n_rot, "semantic": a.semantic,
                         "n_samples": n}}, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()