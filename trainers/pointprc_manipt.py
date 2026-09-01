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
from .manipt_modules import NormalizedAdditiveFusion, ManiPTLoss, calculate_manifold_drift


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
        tokenized_prompts = self.tokenized_prompts

        prompts = self.prompt_learner()
        # --- prompt-adapted (h_*) features ---
        h_txt_raw = self.text_encoder(prompts, tokenized_prompts)          # (C, D)
        h_txt = h_txt_raw / h_txt_raw.norm(dim=-1, keepdim=True)

        h_3d_raw = self.point_encoder(pts) @ self.pc_projection            # (B, D)
        h_3d = h_3d_raw / h_3d_raw.norm(dim=-1, keepdim=True)

        if self.prompt_learner.training:
            # --- frozen (z_*) features ---
            fixed_embeddings = self.prompt_learner.fixed_embeddings        # (C, D), also used as w
            z_txt = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                z_3d_raw = self.prompt_learner.frozen_point_encoder(pts) @ self.pc_projection
                z_3d = z_3d_raw / z_3d_raw.norm(dim=-1, keepdim=True)

            # --- Module B: Normalized Additive Fusion ---
            f_3d = self.fusion(z_3d, h_3d)          # (B, D)
            f_txt = self.fusion(z_txt, h_txt)       # (C, D)

            # --- Module C: classification logits ---
            logits = self.logit_scale.exp() * (f_3d @ f_txt.t())  # (B, C)

            w_text_proto = z_txt   # LLM-enriched semantic prototypes (already normalized)

            return f_3d, z_3d, f_txt, w_text_proto, logits, label
        else:
            # eval: use fused features too (consistent with train-time behavior)
            fixed_embeddings = self.prompt_learner.fixed_embeddings
            z_txt = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                z_3d_raw = self.prompt_learner.frozen_point_encoder(pts) @ self.pc_projection
                z_3d = z_3d_raw / z_3d_raw.norm(dim=-1, keepdim=True)

            f_3d = self.fusion(z_3d, h_3d)
            f_txt = self.fusion(z_txt, h_txt)
            logits = self.logit_scale.exp() * (f_3d @ f_txt.t())
            return logits


@TRAINER_REGISTRY.register()
class PointPRC_ManiPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PointPRC_ManiPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading ULIP (3D Backbone: {cfg.PointEncoder.NAME})")
        # NOTE: load_ulip_to_cpu reads cfg.TRAINER.PointPRC.*; we temporarily
        # bridge the ManiPT config values into that namespace so the shared
        # helper functions from pointprc.py work unmodified.
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

        self.criterion = ManiPTLoss(
            lambda_reg=cfg.TRAINER.PointPRC_ManiPT.LAMBDA_REG,
            tau=cfg.TRAINER.PointPRC_ManiPT.TAU,
        )

        self.scaler = GradScaler() if cfg.TRAINER.PointPRC_ManiPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _build_optim(self, cfg):
        from dassl.optim import build_optimizer
        return build_optimizer(self.model, cfg.OPTIM)

    def _build_sched(self, cfg):
        from dassl.optim import build_lr_scheduler
        return build_lr_scheduler(self.optim, cfg.OPTIM)

    def forward_backward(self, batch):
        pts, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PointPRC_ManiPT.PREC
        if prec == "amp":
            with autocast():
                f_3d, z_3d, f_txt, w_proto, logits, targets = model(pts, label)
                total_loss, l_ce, l_3d, l_txt = self.criterion(
                    f_3d, z_3d, f_txt, w_proto, logits, targets)
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            f_3d, z_3d, f_txt, w_proto, logits, targets = model(pts, label)
            total_loss, l_ce, l_3d, l_txt = self.criterion(
                f_3d, z_3d, f_txt, w_proto, logits, targets)
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        loss_summary = {
            "loss": total_loss.item(),
            "l_ce": l_ce.item(),
            "l_3d": l_3d.item(),
            "l_txt": l_txt.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

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