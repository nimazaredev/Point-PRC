import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint

# reuse identical building blocks from the original trainer
from .pointprc import load_ulip_to_cpu, TextEncoder, VLPromptLearner
from .manipt_modules import (
    NormalizedAdditiveFusion, ManiPTLoss, calculate_manifold_drift,
    sample_rotations, apply_rotation, RotationSubspace, RotationInvarianceLoss,
    sample_scales, apply_scale,
)

class CustomULIP_ManiPT(nn.Module):
    def __init__(self, cfg, classnames, ulip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, ulip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.point_encoder = ulip_model.point_encoder
        self.pc_projection = ulip_model.pc_projection
        self.text_encoder = TextEncoder(ulip_model)
        self.logit_scale = ulip_model.logit_scale
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        self.fusion = NormalizedAdditiveFusion(eps=1e-8)
        tau = cfg.TRAINER.PointPRC_ManiPT.TAU
        self.tau = tau

    def forward(self, pts, label=None):
        """
        `pts` can be a batch of any size (the trainer concatenates the rotated views
        along the batch dimension). Text features do not depend on the views and are
        computed once per call.
        """
        prompts = self.prompt_learner()
        # --- prompt-adapted (h_*) features ---
        h_txt_raw = self.text_encoder(prompts, self.tokenized_prompts)     # (C, D)
        h_txt = h_txt_raw / h_txt_raw.norm(dim=-1, keepdim=True)
 
        h_3d_raw = self.point_encoder(pts) @ self.pc_projection            # (B, D)
        h_3d = h_3d_raw / h_3d_raw.norm(dim=-1, keepdim=True)
 
        # --- frozen (z_*) features, computed on the SAME (possibly rotated) input ---
        fixed_embeddings = self.prompt_learner.fixed_embeddings            # (C, D), also used as w
        z_txt = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            z_3d_raw = self.prompt_learner.frozen_point_encoder(pts) @ self.pc_projection
            z_3d = z_3d_raw / z_3d_raw.norm(dim=-1, keepdim=True)
 
        # --- Normalized Additive Fusion ---
        f_3d = self.fusion(z_3d, h_3d)
        f_txt = self.fusion(z_txt, h_txt)
        logits = self.logit_scale.exp() * (f_3d @ f_txt.t())
 
        if self.prompt_learner.training:
            return {
                "f_3d": f_3d, "z_3d": z_3d, "h_3d": h_3d,
                "f_txt": f_txt, "w_txt": z_txt,
                "logits": logits,
            }
        return logits
 


@TRAINER_REGISTRY.register()
class PointPRC_ManiPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PointPRC_ManiPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
 
        print(f"Loading ULIP (3D Backbone: {cfg.PointEncoder.NAME})")
        # NOTE: load_ulip_to_cpu reads cfg.TRAINER.PointPRC.*; we bridge the ManiPT
        # config values into that namespace (so TRAINER.PointPRC.* given on the command
        # line are overwritten by TRAINER.PointPRC_ManiPT.* here).
        cfg.defrost()
        cfg.TRAINER.PointPRC.PROMPT_DEPTH_POINT = cfg.TRAINER.PointPRC_ManiPT.PROMPT_DEPTH_POINT
        cfg.TRAINER.PointPRC.PROMPT_DEPTH_TEXT = cfg.TRAINER.PointPRC_ManiPT.PROMPT_DEPTH_TEXT
        cfg.TRAINER.PointPRC.N_CTX_POINT = cfg.TRAINER.PointPRC_ManiPT.N_CTX_POINT
        cfg.TRAINER.PointPRC.N_CTX_TEXT = cfg.TRAINER.PointPRC_ManiPT.N_CTX_TEXT
        cfg.TRAINER.PointPRC.CTX_INIT = cfg.TRAINER.PointPRC_ManiPT.CTX_INIT
        cfg.TRAINER.PointPRC.PROMPT_TYPE = cfg.TRAINER.PointPRC_ManiPT.PROMPT_TYPE
        cfg.TRAINER.PointPRC.NO_TDC = cfg.TRAINER.PointPRC_ManiPT.NO_TDC
        cfg.freeze()
 
        ulip_model = load_ulip_to_cpu(cfg)
 
        if cfg.TRAINER.PointPRC_ManiPT.PREC in ["fp32", "amp"]:
            ulip_model.float()
 
        print("Building CustomULIP_ManiPT")
        self.model = CustomULIP_ManiPT(cfg, classnames, ulip_model)
 
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('\n', '=' * 30, '\n >>> Learnable Params:', params, '\n', '=' * 30, '\n')
 
        enabled_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Parameters to be updated: {enabled_names}")
 
        self.model.to(self.device)
        self.optim = self._build_optim(cfg)
        self.sched = self._build_sched(cfg)
        self.register_model("pointprc_manipt", self.model, self.optim, self.sched)
 
        rc = cfg.TRAINER.PointPRC_ManiPT
        self.criterion = ManiPTLoss(lambda_reg=rc.LAMBDA_REG, tau=rc.TAU)
 
        self._build_rotation_parts(cfg)   # [NEW - ADD] <-- the only new line inside build_model
 
        self.scaler = GradScaler() if rc.PREC == "amp" else None
 
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
    
    def _build_rotation_parts(self, cfg):
        rc = cfg.TRAINER.PointPRC_ManiPT
        self.rot_mode = rc.ROT_MODE
        self.gamma_s = float(rc.GAMMA_S)
        self.gamma_inv = float(rc.GAMMA_INV)
        self.anchor_clean = bool(rc.ANCHOR_CLEAN)

        if self.rot_mode == "none" and (self.gamma_s > 0 or self.gamma_inv > 0):
            raise ValueError("GAMMA_S / GAMMA_INV > 0 requires ROT_MODE != 'none' "
                             "(both losses are defined on rotated views).")
 
        # dedicated CPU generators: the SAME rotations in every experiment for a given seed,
        # regardless of which loss is on (fair comparison E1..E4)
        seed = int(cfg.SEED) if cfg.SEED >= 0 else 0
        self.rot_gen = torch.Generator().manual_seed(seed + 12345)
 
        self.sens_loss = None
        if self.gamma_s > 0:
            if not rc.SUBSPACE_PATH:
                raise ValueError("GAMMA_S > 0 requires TRAINER.PointPRC_ManiPT.SUBSPACE_PATH "
                                 "(run compute_rot_subspace.py first).")
            self.sens_loss = RotationSubspace(
                rc.SUBSPACE_PATH, mode=rc.SUBSPACE_MODE,
                k=rc.SUBSPACE_K, energy=rc.SUBSPACE_ENERGY).to(self.device)
            print(f"[Rot] train mode={self.rot_mode} (max {rc.ROT_MAX_DEG} deg, axis {rc.ROT_AXIS}, "
              f"scale [{rc.SCALE_MIN}, {rc.SCALE_MAX}]), ")
 
        self.inv_loss = None
        if self.gamma_inv > 0:
            self.inv_loss = RotationInvarianceLoss(rc.INV_KIND, rc.INV_STOPGRAD)
 
        # test-time rotation (evaluation on rotated test data)
        self.test_rot_mode = rc.TEST_ROT_MODE
        self.test_gen = torch.Generator()
        self._test_split = None
        print(f"[Rot] train mode={self.rot_mode} (max {rc.ROT_MAX_DEG} deg, axis {rc.ROT_AXIS}), "
              f"gamma_s={self.gamma_s}, gamma_inv={self.gamma_inv}, test mode={self.test_rot_mode}")
    
    
    def _make_views(self, pts):
        """Returns a list of views. 'none' -> [pts]; otherwise exactly 2 rotated views."""
        rc = self.cfg.TRAINER.PointPRC_ManiPT
        if self.rot_mode == "none":
            return [pts]
        views = []
        for j in range(2):
            if j == 0 and rc.ROT_FIRST_IDENTITY:
                views.append(pts)
                continue
            R = sample_rotations(pts.shape[0], self.rot_mode, rc.ROT_MAX_DEG,
                                 rc.ROT_AXIS, self.rot_gen, pts.device)
            v = apply_rotation(pts, R)
            # optional random scaling (off when SCALE_MIN = SCALE_MAX = 1.0)
            S = sample_scales(pts.shape[0], rc.SCALE_MIN, rc.SCALE_MAX, self.rot_gen, pts.device)
            views.append(apply_scale(v, S))
        return views
 
    def _gamma_s_now(self):
        rc = self.cfg.TRAINER.PointPRC_ManiPT
        if rc.GAMMA_S_SCHEDULE == "linear_decay":
            t = self.epoch / max(1, self.max_epoch - 1)
            return self.gamma_s * max(0.0, 1.0 - t)
        return self.gamma_s
 
    def _compute_loss(self, pts, label):
        views = self._make_views(pts)
        V = len(views)
        x = torch.cat(views, dim=0) if V > 1 else views[0]
        y = label.repeat(V) if V > 1 else label
 
        out = self.model(x, y)
        f3 = out["f_3d"].chunk(V)
        z3 = out["z_3d"].chunk(V)
        h3 = out["h_3d"].chunk(V)
        lg = out["logits"].chunk(V)
        yy = y.chunk(V)

        # canonical anchor: frozen feature of the ORIGINAL (unrotated) input
        z_clean = None
        if self.anchor_clean:
            with torch.no_grad():
                m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                zc = m.prompt_learner.frozen_point_encoder(pts) @ m.pc_projection
                z_clean = zc / zc.norm(dim=-1, keepdim=True)          # (B, D)

        # L_aug = mean_j [CE_j + lam (L3D_j + Ltxt)]   (same views for E1..E4)
        tot, ce, l3d, ltxt = 0.0, 0.0, 0.0, 0.0
        for j in range(V):
            anchor = z_clean if z_clean is not None else z3[j]
            t, c, a, b = self.criterion(f3[j], anchor, out["f_txt"], out["w_txt"], lg[j], yy[j])
            tot, ce, l3d, ltxt = tot + t, ce + c, l3d + a, ltxt + b
        total = tot / V
        summary = {"l_ce": ce.item() / V, "l_3d": l3d.item() / V, "l_txt": ltxt.item() / V}
 
        # idea 1: penalty on drift in rotation-sensitive directions
        if self.sens_loss is not None:
            l_sens = sum(self.sens_loss(h3[j], z3[j]) for j in range(V)) / V
            g = self._gamma_s_now()
            total = total + g * l_sens
            summary["l_sens"] = l_sens.item()
 
        # idea 2: invariance between the two views
        if self.inv_loss is not None:
            l_inv = self.inv_loss(h3[0], h3[1], f3[0], f3[1], lg[0], lg[1])
            total = total + self.gamma_inv * l_inv
            summary["l_inv"] = l_inv.item()
 
        summary["loss"] = total.item()
        return total, summary
    
    def _build_optim(self, cfg):
        from dassl.optim import build_optimizer
        return build_optimizer(self.model, cfg.OPTIM)

    def _build_sched(self, cfg):
        from dassl.optim import build_lr_scheduler
        return build_lr_scheduler(self.optim, cfg.OPTIM)

    def forward_backward(self, batch):
        pts, label = self.parse_batch_train(batch)
        optim = self.optim
        scaler = self.scaler
 
        if self.cfg.TRAINER.PointPRC_ManiPT.PREC == "amp":
            with autocast():
                total, summary = self._compute_loss(pts, label)
            optim.zero_grad()
            scaler.scale(total).backward()
            scaler.step(optim)
            scaler.update()
        else:
            total, summary = self._compute_loss(pts, label)
            optim.zero_grad()
            total.backward()
            optim.step()
 
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
 
        return summary

    def test(self, split=None):
        self._test_split = split
        if self.test_rot_mode != "none":
            rc = self.cfg.TRAINER.PointPRC_ManiPT
            self.test_gen.manual_seed(int(rc.TEST_ROT_SEED))   # same rotations every call
        return super().test(split)
 
    def model_inference(self, input):
        if self.test_rot_mode != "none" and self._test_split in (None, "test"):
            rc = self.cfg.TRAINER.PointPRC_ManiPT
            R = sample_rotations(input.shape[0], self.test_rot_mode, rc.TEST_ROT_MAX_DEG,
                                 rc.TEST_ROT_AXIS, self.test_gen, input.device)
            input = apply_rotation(input, R)
        return self.model(input)

    def parse_batch_train(self, batch):
        input = batch["pc"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_ = checkpoint["epoch"]

            state_dict.pop("prompt_learner.token_prefix", None)
            state_dict.pop("prompt_learner.token_suffix", None)

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_})')
            self._models[name].load_state_dict(state_dict, strict=False)