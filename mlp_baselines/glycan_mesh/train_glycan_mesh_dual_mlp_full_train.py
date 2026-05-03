#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_glycan_mesh_dual_mlp_full_train.py
========================================

Full-data single-model training for glycan+MeSH dual-MLP multi-label prediction.

Architecture
------------
Dual encoder:
  glycan embedding --MLP--> joint space
  MeSH embedding   --MLP--> joint space
  score(glycan, mesh) = scaled dot product in the joint space

Training objective
------------------
Supervised multi-label BCE over the FULL candidate MeSH space.

This is the full-train counterpart of:
  optuna_glycan_mesh_dual_mlp_holdout_test_cv5.py

Outputs (under --out_dir/full_train)
------------------------------------
  final_ckpt_stage1.pth
  final_ckpt_stage2.pth
  final_ckpt_full.pth
  history.csv
  train_summary.json

Example
-------
python train_glycan_mesh_dual_mlp_full_train.py \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
  --out_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/final_full_model_run01 \
  --proj_dim 768 \
  --gly_hidden1 1024 \
  --gly_hidden2 0 \
  --mesh_hidden1 1536 \
  --mesh_hidden2 0 \
  --dropout 0.07011072808496752 \
  --normalize_joint \
  --init_logit_scale 17.823814197845874 \
  --lr_stage1 0.00015212322552301635 \
  --lr_stage2 0.0002908364586848232 \
  --weight_decay 4.097015818680126e-05 \
  --batch_size 128 \
  --mesh_batch_size 4096 \
  --pos_weight_mode balanced \
  --pos_weight_clip_max 20.0 \
  --epochs_stage1 50 \
  --epochs_stage2 60 \
  --device cuda \
  --seed 42 \
  --num_workers 0 \
  --grad_clip 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ----------------------------
# 1) Reproducibility / utils
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _coerce_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ----------------------------
# 2) Dataset
# ----------------------------
class GlycanIndexDataset(Dataset):
    def __init__(self, gly_indices: List[int]):
        self.gly_indices = sorted(set(int(x) for x in gly_indices))
        if len(self.gly_indices) == 0:
            raise ValueError("Empty glycan index list.")

    def __len__(self) -> int:
        return len(self.gly_indices)

    def __getitem__(self, i: int) -> int:
        return self.gly_indices[i]


# ----------------------------
# 3) Model
# ----------------------------
def _make_mlp(in_dim: int, hidden1: int, hidden2: int, out_dim: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim

    if hidden1 > 0:
        layers.extend([
            nn.Linear(prev, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        prev = hidden1

    if hidden2 > 0:
        layers.extend([
            nn.Linear(prev, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        prev = hidden2

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GlyMeshDualMLP(nn.Module):
    """
    Dual encoder:
      glycan embedding -> MLP -> z_g
      mesh embedding   -> MLP -> z_m
      score = scale * z_g @ z_m^T
    """

    def __init__(
        self,
        gly_in_dim: int,
        mesh_in_dim: int,
        proj_dim: int = 512,
        gly_hidden1: int = 1024,
        gly_hidden2: int = 0,
        mesh_hidden1: int = 1024,
        mesh_hidden2: int = 0,
        dropout: float = 0.1,
        normalize_joint: bool = True,
        init_logit_scale: float = 1.0,
    ):
        super().__init__()
        self.gly_net = _make_mlp(gly_in_dim, gly_hidden1, gly_hidden2, proj_dim, dropout)
        self.mesh_net = _make_mlp(mesh_in_dim, mesh_hidden1, mesh_hidden2, proj_dim, dropout)
        self.normalize_joint = bool(normalize_joint)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(math.log(init_logit_scale)), dtype=torch.float32)
        )

    def encode_gly(self, x_g: torch.Tensor) -> torch.Tensor:
        z = self.gly_net(x_g)
        if self.normalize_joint:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def encode_mesh(self, x_m: torch.Tensor) -> torch.Tensor:
        z = self.mesh_net(x_m)
        if self.normalize_joint:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def score_from_encoded(self, z_g: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1e-3, max=100.0)
        return scale * (z_g @ z_m.T)

    def forward_scores(self, x_g: torch.Tensor, x_m: torch.Tensor) -> torch.Tensor:
        z_g = self.encode_gly(x_g)
        z_m = self.encode_mesh(x_m)
        return self.score_from_encoded(z_g, z_m)


# ----------------------------
# 4) IO helpers
# ----------------------------
def load_embeddings_from_csv(
    gly_emb_csv: str,
    mesh_emb_csv: str,
    normalize_gly: bool = True,
    normalize_mesh: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], Dict[str, int], Dict[str, int]]:
    gly_df = pd.read_csv(gly_emb_csv)
    mesh_df = pd.read_csv(mesh_emb_csv)

    gly_id_col = gly_df.columns[0]
    mesh_id_col = mesh_df.columns[0]

    gly_ids = gly_df[gly_id_col].astype(str).tolist()
    mesh_ids = mesh_df[mesh_id_col].astype(str).tolist()

    gly_id2row = {gid: i for i, gid in enumerate(gly_ids)}
    mesh_id2row = {mid: i for i, mid in enumerate(mesh_ids)}

    G_np = gly_df.drop(columns=[gly_id_col]).to_numpy(dtype=np.float32)
    M_np = mesh_df.drop(columns=[mesh_id_col]).to_numpy(dtype=np.float32)

    G = torch.from_numpy(G_np)
    M = torch.from_numpy(M_np)

    if normalize_gly:
        G = l2_normalize(G)
    if normalize_mesh:
        M = l2_normalize(M)

    return G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row


def build_pairs_from_csv(
    pairs_csv: str,
    gly_id2row: Dict[str, int],
    mesh_id2row: Dict[str, int],
    gly_id_col: str = "glytoucan_ac",
    mesh_list_col: str = "descriptor_ui_list",
    mesh_sep: str = ";",
) -> Tuple[List[Tuple[int, int]], int, int]:
    pairs_df = pd.read_csv(pairs_csv)

    pairs: List[Tuple[int, int]] = []
    missing_gly = 0
    missing_mesh = 0

    for _, row in pairs_df.iterrows():
        gly_id = row.get(gly_id_col, None)
        mesh_list = row.get(mesh_list_col, None)
        if pd.isna(gly_id) or pd.isna(mesh_list):
            continue

        gi = gly_id2row.get(str(gly_id), None)
        if gi is None:
            missing_gly += 1
            continue

        mids = [s.strip() for s in str(mesh_list).split(mesh_sep) if s.strip()]
        for mid in mids:
            mi = mesh_id2row.get(mid, None)
            if mi is None:
                missing_mesh += 1
                continue
            pairs.append((gi, mi))

    pairs = sorted(set(pairs))
    return pairs, missing_gly, missing_mesh


def build_multilabel_targets(
    pairs: List[Tuple[int, int]],
    n_gly: int,
    n_mesh: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[int, Set[int]]]:
    Y = torch.zeros((n_gly, n_mesh), dtype=dtype)
    g2true: Dict[int, Set[int]] = defaultdict(set)

    for g, m in pairs:
        Y[g, m] = 1.0
        g2true[g].add(m)

    return Y, g2true


# ----------------------------
# 5) Training helpers
# ----------------------------
def make_pos_weight(
    Y_train: torch.Tensor,
    mode: str = "none",
    clip_min: float = 1.0,
    clip_max: float = 100.0,
) -> Optional[torch.Tensor]:
    mode = str(mode).lower()
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown pos_weight mode: {mode}")

    pos = Y_train.sum(dim=0)
    total = torch.tensor(float(Y_train.size(0)))
    neg = total - pos
    pw = neg / torch.clamp(pos, min=1.0)
    pw = torch.clamp(pw, min=float(clip_min), max=float(clip_max))
    return pw.to(dtype=torch.float32)


def _save_ckpt(path: Path, model: nn.Module, cfg_dict: Dict[str, Any], extra: Dict[str, Any] | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "cfg": cfg_dict,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def _compute_logits_full(
    model: GlyMeshDualMLP,
    x_g: torch.Tensor,
    M: torch.Tensor,
    device: torch.device,
    mesh_batch_size: int,
) -> torch.Tensor:
    """
    Compute logits against the FULL candidate MeSH space by chunking mesh embeddings.
    Returns: [B, n_mesh]
    """
    z_g = model.encode_gly(x_g)
    scale = model.logit_scale.exp().clamp(min=1e-3, max=100.0)

    logits_parts: List[torch.Tensor] = []
    for s in range(0, M.size(0), mesh_batch_size):
        e = min(M.size(0), s + mesh_batch_size)
        z_m = model.encode_mesh(M[s:e].to(device, non_blocking=True))
        logits_parts.append(scale * (z_g @ z_m.T))

    return torch.cat(logits_parts, dim=1)


# ----------------------------
# 6) Full training
# ----------------------------
def train(cfg) -> None:
    set_seed(int(cfg.seed))
    device = _get_device(cfg.device)
    print("Device:", device)

    out_dir = Path(cfg.out_dir)
    run_dir = out_dir / "full_train"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- load embeddings ----
    G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row = load_embeddings_from_csv(
        cfg.gly_emb_csv,
        cfg.mesh_emb_csv,
        normalize_gly=(not cfg.no_l2_norm_gly),
        normalize_mesh=(not cfg.no_l2_norm_mesh),
    )

    # ---- supervision pairs ----
    pairs, missing_gly, missing_mesh = build_pairs_from_csv(
        pairs_csv=cfg.pairs_csv,
        gly_id2row=gly_id2row,
        mesh_id2row=mesh_id2row,
        gly_id_col=cfg.gly_id_col,
        mesh_list_col=cfg.mesh_list_col,
        mesh_sep=cfg.mesh_sep,
    )
    if not pairs:
        raise ValueError("No valid pairs after ID mapping.")

    print(
        f"[pairs] built={len(pairs)} "
        f"missing_gly={missing_gly} "
        f"missing_mesh={missing_mesh} "
        f"n_mesh_labels(full_candidate)={len(mesh_ids)}"
    )

    # ---- dense multi-label targets ----
    Y, _ = build_multilabel_targets(
        pairs=pairs,
        n_gly=G.size(0),
        n_mesh=len(mesh_ids),
    )

    train_gly = sorted({g for (g, _) in pairs})
    ds_train = GlycanIndexDataset(train_gly)
    dl = DataLoader(
        ds_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = GlyMeshDualMLP(
        gly_in_dim=G.size(1),
        mesh_in_dim=M.size(1),
        proj_dim=int(cfg.proj_dim),
        gly_hidden1=int(cfg.gly_hidden1),
        gly_hidden2=int(cfg.gly_hidden2),
        mesh_hidden1=int(cfg.mesh_hidden1),
        mesh_hidden2=int(cfg.mesh_hidden2),
        dropout=float(cfg.dropout),
        normalize_joint=bool(cfg.normalize_joint),
        init_logit_scale=float(cfg.init_logit_scale),
    ).to(device)

    Y_train = Y[train_gly]
    pos_weight = make_pos_weight(
        Y_train=Y_train,
        mode=str(cfg.pos_weight_mode),
        clip_max=float(cfg.pos_weight_clip_max),
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=(pos_weight.to(device) if pos_weight is not None else None)
    )

    history_rows: List[Dict[str, Any]] = []
    cfg_dict = _coerce_jsonable(vars(cfg).copy())

    def train_stage(stage: int, lr: float, n_epochs: int, ckpt_name: str) -> None:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(cfg.weight_decay),
        )

        for ep in range(1, int(n_epochs) + 1):
            model.train()
            loss_sum = 0.0
            n_batches = 0
            n_seen = 0
            loss_ema = None

            pbar = tqdm(dl, desc=f"full_train stage{stage} epoch {ep}/{n_epochs}", leave=True)
            for g_idx in pbar:
                x_g = G[g_idx].to(device, non_blocking=True)
                y = Y[g_idx].to(device, non_blocking=True)

                logits = _compute_logits_full(
                    model=model,
                    x_g=x_g,
                    M=M,
                    device=device,
                    mesh_batch_size=int(cfg.mesh_batch_size),
                )
                loss = criterion(logits, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(cfg.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()

                loss_item = float(loss.item())
                loss_sum += loss_item
                n_batches += 1
                n_seen += int(x_g.size(0))
                loss_ema = loss_item if loss_ema is None else (loss_ema * 0.9 + loss_item * 0.1)

                pbar.set_postfix({
                    "loss_mean": f"{loss_sum / max(n_batches, 1):.4f}",
                    "loss_ema": f"{loss_ema:.4f}",
                })

            prev_epochs = sum(int(r["epoch_in_stage"]) for r in history_rows if int(r["stage"]) < stage)
            row = {
                "stage": int(stage),
                "epoch_in_stage": int(ep),
                "global_epoch": int(prev_epochs + ep),
                "lr": float(lr),
                "train_loss_mean": float(loss_sum / max(n_batches, 1)),
                "train_loss_ema": float(loss_ema if loss_ema is not None else float("nan")),
                "n_batches": int(n_batches),
                "n_seen_glycans": int(n_seen),
                "logit_scale_exp": float(model.logit_scale.exp().clamp(min=1e-3, max=100.0).detach().cpu().item()),
            }
            history_rows.append(row)

            print(
                f"[full_train] stage={stage} epoch={ep}/{n_epochs} "
                f"loss_mean={row['train_loss_mean']:.6f} "
                f"loss_ema={row['train_loss_ema']:.6f} "
                f"logit_scale={row['logit_scale_exp']:.6f}"
            )

        _save_ckpt(
            run_dir / ckpt_name,
            model=model,
            cfg_dict=cfg_dict,
            extra={
                "stage": int(stage),
                "history_last": history_rows[-1] if history_rows else None,
                "n_train_pairs": int(len(pairs)),
                "n_train_glycans": int(len(train_gly)),
                "n_mesh_labels": int(len(mesh_ids)),
            },
        )

    # stage 1
    train_stage(
        stage=1,
        lr=float(cfg.lr_stage1),
        n_epochs=int(cfg.epochs_stage1),
        ckpt_name="final_ckpt_stage1.pth",
    )

    # stage 2
    print("[full_train] Stage2 init: continue from Stage1 LAST weights.")
    train_stage(
        stage=2,
        lr=float(cfg.lr_stage2),
        n_epochs=int(cfg.epochs_stage2),
        ckpt_name="final_ckpt_stage2.pth",
    )

    # alias final full checkpoint
    shutil.copy2(run_dir / "final_ckpt_stage2.pth", run_dir / "final_ckpt_full.pth")

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(run_dir / "history.csv", index=False)

    summary = {
        "model": "glycan_mesh_dual_mlp_multilabel_full_train",
        "gly_emb_csv": str(cfg.gly_emb_csv),
        "mesh_emb_csv": str(cfg.mesh_emb_csv),
        "pairs_csv": str(cfg.pairs_csv),
        "n_pairs_total": int(len(pairs)),
        "n_unique_glycans_train": int(len(train_gly)),
        "n_mesh_labels": int(len(mesh_ids)),
        "missing_gly": int(missing_gly),
        "missing_mesh": int(missing_mesh),
        "train_params": {
            "proj_dim": int(cfg.proj_dim),
            "gly_hidden1": int(cfg.gly_hidden1),
            "gly_hidden2": int(cfg.gly_hidden2),
            "mesh_hidden1": int(cfg.mesh_hidden1),
            "mesh_hidden2": int(cfg.mesh_hidden2),
            "dropout": float(cfg.dropout),
            "normalize_joint": bool(cfg.normalize_joint),
            "init_logit_scale": float(cfg.init_logit_scale),
            "lr_stage1": float(cfg.lr_stage1),
            "lr_stage2": float(cfg.lr_stage2),
            "weight_decay": float(cfg.weight_decay),
            "epochs_stage1": int(cfg.epochs_stage1),
            "epochs_stage2": int(cfg.epochs_stage2),
            "batch_size": int(cfg.batch_size),
            "mesh_batch_size": int(cfg.mesh_batch_size),
            "pos_weight_mode": str(cfg.pos_weight_mode),
            "pos_weight_clip_max": float(cfg.pos_weight_clip_max),
            "grad_clip": float(cfg.grad_clip),
            "seed": int(cfg.seed),
            "device": str(device),
            "no_l2_norm_gly": bool(cfg.no_l2_norm_gly),
            "no_l2_norm_mesh": bool(cfg.no_l2_norm_mesh),
        },
        "artifacts": {
            "history_csv": str((run_dir / "history.csv").as_posix()),
            "stage1_ckpt": str((run_dir / "final_ckpt_stage1.pth").as_posix()),
            "stage2_ckpt": str((run_dir / "final_ckpt_stage2.pth").as_posix()),
            "full_ckpt": str((run_dir / "final_ckpt_full.pth").as_posix()),
        },
    }
    (run_dir / "train_summary.json").write_text(json.dumps(_coerce_jsonable(summary), indent=2), encoding="utf-8")

    print("[full_train] wrote:", (run_dir / "history.csv").as_posix())
    print("[full_train] wrote:", (run_dir / "train_summary.json").as_posix())
    print("[full_train] final checkpoint:", (run_dir / "final_ckpt_full.pth").as_posix())


# ----------------------------
# 7) CLI
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--mesh_emb_csv", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")
    ap.add_argument("--no_l2_norm_gly", action="store_true", help="Disable L2 normalization of input glycan embeddings")
    ap.add_argument("--no_l2_norm_mesh", action="store_true", help="Disable L2 normalization of input MeSH embeddings")

    # model
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--gly_hidden1", type=int, default=1024)
    ap.add_argument("--gly_hidden2", type=int, default=0)
    ap.add_argument("--mesh_hidden1", type=int, default=1024)
    ap.add_argument("--mesh_hidden2", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--normalize_joint", action="store_true", help="L2-normalize projected glycan/MeSH embeddings before scoring")
    ap.add_argument("--init_logit_scale", type=float, default=1.0)

    # training
    ap.add_argument("--lr_stage1", type=float, default=1e-3)
    ap.add_argument("--lr_stage2", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs_stage1", type=int, default=50)
    ap.add_argument("--epochs_stage2", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--mesh_batch_size", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # imbalance handling
    ap.add_argument("--pos_weight_mode", type=str, default="balanced", choices=["none", "balanced"])
    ap.add_argument("--pos_weight_clip_max", type=float, default=50.0)

    # misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", required=True)

    return ap


if __name__ == "__main__":
    cfg = build_argparser().parse_args()
    train(cfg)