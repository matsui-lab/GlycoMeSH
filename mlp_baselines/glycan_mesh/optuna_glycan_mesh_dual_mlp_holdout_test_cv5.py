#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
optuna_glycan_mesh_dual_mlp_holdout_test_cv5.py
================================================

Dual-encoder MLP baseline:
  glycan embedding --MLP--> joint space
  MeSH embedding   --MLP--> joint space
  score(glycan, mesh) = scaled dot product in the joint space

Training objective:
  supervised multi-label BCE over the FULL candidate MeSH space.

This script is intended as a practical extension of the glycan-only MLP baseline,
so that both glycan and MeSH embeddings are used while preserving the same
holdout TEST + DEV CV + Optuna protocol.

Strategy:
  1) Fix glycan-level TEST split ONCE
  2) On DEV, run Optuna with 5-fold CV mean(metric)
  3) Select best HP
  4) Retrain on ALL DEV (no val)
  5) Final evaluation on fixed TEST

Outputs (under --study_dir):
  study.db
  study_best.json
  split_test_glycans.json
  fold_assignment_dev.csv
  trials/trial_<number>/
    params.json
    cv_summary.csv
    fold_0/ ... fold_4/
  final_dev_train/
    final_ckpt_stage1.pth
    final_ckpt_stage2.pth
    test_metrics.json

Notes
-----
- This is NOT the contrastive StableRep objective.
- This is also NOT a fully explicit pairwise concat-MLP over every glycan-mesh pair,
  because that becomes memory-heavy over the full candidate MeSH space.
- Instead, this uses two MLP towers and a supervised BCE ranking loss over all MeSH labels.
- The evaluation protocol matches the glycan-only MLP script closely.
"""

from __future__ import annotations

import argparse
import json
import time
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import optuna
from optuna.trial import Trial
from sklearn.model_selection import KFold


# ----------------------------
# 1) Reproducibility / math
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


def _parse_metric_k(metric: str) -> Optional[int]:
    m = re.match(r"^(hit|recall|precision|mrr)@(\d+)$", metric.strip())
    if not m:
        return None
    return int(m.group(2))


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

    Optionally L2-normalizes the projected embeddings before scoring.
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
        self.logit_scale = nn.Parameter(torch.tensor(float(math.log(init_logit_scale)), dtype=torch.float32))

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


def make_kfold_assignment_on_unique_glycans(
    pairs: List[Tuple[int, int]],
    n_folds: int,
    seed: int,
) -> Tuple[Dict[int, int], List[int]]:
    gly_in_pairs = sorted({g for (g, _) in pairs})
    if len(gly_in_pairs) < n_folds:
        raise ValueError(
            f"Not enough unique glycans for n_folds={n_folds}. unique_glycans_in_pairs={len(gly_in_pairs)}"
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    row2fold: Dict[int, int] = {}
    for fold_id, (_, val_idx) in enumerate(kf.split(gly_in_pairs)):
        for j in val_idx:
            row2fold[gly_in_pairs[j]] = int(fold_id)

    if len(row2fold) != len(gly_in_pairs):
        raise RuntimeError("KFold assignment failed: some glycans did not receive a fold id.")

    return row2fold, gly_in_pairs


def make_fixed_test_split(
    *,
    pairs: List[Tuple[int, int]],
    test_frac: float,
    seed: int,
) -> Tuple[Set[int], Set[int]]:
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"--test_frac must be in (0,1), got {test_frac}")

    gly_in_pairs = sorted({g for (g, _) in pairs})
    if len(gly_in_pairs) < 10:
        raise ValueError(f"Too few unique glycans in pairs: {len(gly_in_pairs)}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(gly_in_pairs))
    n_test = int(round(len(gly_in_pairs) * float(test_frac)))
    n_test = max(1, n_test)
    n_test = min(len(gly_in_pairs) - 1, n_test)

    test_idx = set(int(i) for i in perm[:n_test].tolist())
    test_g = set(gly_in_pairs[i] for i in test_idx)
    dev_g = set(gly_in_pairs) - test_g
    return test_g, dev_g


def filter_pairs_by_glycans(pairs: List[Tuple[int, int]], keep_glycans: Set[int]) -> List[Tuple[int, int]]:
    return [(g, m) for (g, m) in pairs if g in keep_glycans]


# ----------------------------
# 5) Evaluation
# ----------------------------
@torch.no_grad()
def eval_dual_ranking(
    model: GlyMeshDualMLP,
    G: torch.Tensor,
    M: torch.Tensor,
    g2true: Dict[int, Set[int]],
    device: torch.device,
    batch_size_gly: int = 512,
    batch_size_mesh: int = 2048,
    topks: List[int] | Tuple[int, ...] = (1, 10, 15, 20, 25, 30),
) -> Dict[str, float]:
    model.eval()

    topks = sorted(set(int(k) for k in topks))
    max_k = max(topks)

    zm_parts = []
    for s in range(0, M.size(0), batch_size_mesh):
        e = min(M.size(0), s + batch_size_mesh)
        zm_parts.append(model.encode_mesh(M[s:e].to(device)).cpu())
    zm_all = torch.cat(zm_parts, dim=0)
    zmT = zm_all.T.contiguous()
    scale = model.logit_scale.exp().clamp(min=1e-3, max=100.0).detach().cpu()

    hit = {k: 0 for k in topks}
    rec = {k: 0.0 for k in topks}
    prec = {k: 0.0 for k in topks}
    mrr = {k: 0.0 for k in topks}
    n = 0

    eval_gly = sorted(g2true.keys())
    for s in range(0, len(eval_gly), batch_size_gly):
        batch_g_idx = eval_gly[s:s + batch_size_gly]
        z_g = model.encode_gly(G[batch_g_idx].to(device)).cpu()
        logits = scale * (z_g @ zmT)

        for local_i, g_idx in enumerate(batch_g_idx):
            scores = logits[local_i]
            top_all = torch.topk(scores, k=max_k).indices.tolist()
            true_set = set(g2true[g_idx])
            denom_true = max(1, len(true_set))
            n += 1

            for k in topks:
                topk = top_all[:k]
                if any(m in true_set for m in topk):
                    hit[k] += 1

                inter = len(set(topk).intersection(true_set))
                rec[k] += inter / denom_true
                prec[k] += inter / float(k)

                rr = 0.0
                for rank, midx in enumerate(topk, start=1):
                    if midx in true_set:
                        rr = 1.0 / rank
                        break
                mrr[k] += rr

    denom = max(1, n)
    metrics: Dict[str, float] = {"n_eval": float(n)}
    for k in topks:
        metrics[f"hit@{k}"] = hit[k] / denom
        metrics[f"recall@{k}"] = rec[k] / denom
        metrics[f"precision@{k}"] = prec[k] / denom
        metrics[f"mrr@{k}"] = mrr[k] / denom
    return metrics


# ----------------------------
# 6) Training helpers
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


def _save_ckpt(
    out_path: Path,
    model: GlyMeshDualMLP,
    cfg,
    fold: int,
    stage: int,
    best_val: float,
    best_epoch_global: int,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "config": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg),
        "fold": fold,
        "stage": stage,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch_global),
    }
    torch.save(ckpt, out_path)


def _load_ckpt(ckpt_path: Path, model: GlyMeshDualMLP, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def _compute_logits_full(
    model: GlyMeshDualMLP,
    x_g: torch.Tensor,
    M: torch.Tensor,
    device: torch.device,
    mesh_batch_size: int,
) -> torch.Tensor:
    z_g = model.encode_gly(x_g)
    scale = model.logit_scale.exp().clamp(min=1e-3, max=100.0)
    logits_parts: List[torch.Tensor] = []

    for s in range(0, M.size(0), mesh_batch_size):
        e = min(M.size(0), s + mesh_batch_size)
        z_m = model.encode_mesh(M[s:e].to(device))
        logits_parts.append(scale * (z_g @ z_m.T))

    return torch.cat(logits_parts, dim=1)


def _train_one_epoch(
    model: GlyMeshDualMLP,
    dl: DataLoader,
    G: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    mesh_batch_size: int,
    desc: str,
) -> float:
    model.train()
    loss_ema = 0.0

    pbar = tqdm(dl, desc=desc, leave=True)
    for g_idx in pbar:
        x_g = G[g_idx].to(device)
        y = Y[g_idx].to(device)

        logits = _compute_logits_full(
            model=model,
            x_g=x_g,
            M=M,
            device=device,
            mesh_batch_size=mesh_batch_size,
        )
        loss = criterion(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_ema = loss_ema * 0.9 + float(loss.item()) * 0.1
        pbar.set_postfix({"loss": f"{loss_ema:.4f}"})

    return float(loss_ema)


def run_stage(
    *,
    stage: int,
    model: GlyMeshDualMLP,
    dl: DataLoader,
    G: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    g2true_val: Optional[Dict[int, Set[int]]],
    device: torch.device,
    fold: int,
    cfg,
    fold_dir: Path,
    start_epoch_global: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: Optional[torch.Tensor],
    key_best: Optional[str],
    key_es: Optional[str],
    eval_topks: Optional[List[int]],
    early_stop: bool,
    patience: int,
    min_delta: float,
    warmup_epochs_global: int,
    ckpt_name: str,
) -> Tuple[List[Dict[str, float]], float, int, int, Optional[Path]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(pos_weight.to(device) if pos_weight is not None else None)
    )

    history_rows: List[Dict[str, float]] = []
    best_val = -1.0
    best_epoch_global = -1
    best_ckpt_path: Optional[Path] = None

    patience_left = int(patience)
    last_best_for_es = -1e18

    eval_every = int(getattr(cfg, "eval_every", 0))
    do_eval = (
        (g2true_val is not None)
        and (len(g2true_val) > 0)
        and (eval_topks is not None)
        and (eval_every > 0)
        and (key_best is not None)
        and (key_es is not None)
    )

    for i in range(1, n_epochs + 1):
        epoch_global = start_epoch_global + i

        loss_ema = _train_one_epoch(
            model=model,
            dl=dl,
            G=G,
            M=M,
            Y=Y,
            device=device,
            opt=opt,
            criterion=criterion,
            grad_clip=float(cfg.grad_clip),
            mesh_batch_size=int(cfg.mesh_batch_size),
            desc=f"fold {fold} stage{stage} epoch {epoch_global}",
        )
        print(f"[fold {fold} S{stage} E{epoch_global}] train loss (EMA) = {loss_ema:.4f}")

        row: Dict[str, float] = {
            "fold": float(fold),
            "stage": float(stage),
            "epoch": float(epoch_global),
            "epoch_in_stage": float(i),
            "lr": float(lr),
            "train_loss_ema": float(loss_ema),
        }

        evaluated = do_eval and (epoch_global % eval_every == 0)
        if evaluated:
            metrics = eval_dual_ranking(
                model=model,
                G=G,
                M=M,
                g2true=g2true_val,
                device=device,
                batch_size_gly=int(cfg.eval_batch_gly),
                batch_size_mesh=int(cfg.eval_batch_mesh),
                topks=eval_topks,
            )
            row.update(metrics)

            hits = " ".join([f"hit@{k}={metrics[f'hit@{k}']:.4f}" for k in eval_topks])
            print(
                f"  ↳ val: {hits} "
                + f"{key_best}={metrics.get(key_best, float('nan')):.4f} "
                + f"{key_es}={metrics.get(key_es, float('nan')):.4f} "
                + f"n={int(metrics['n_eval'])}"
            )

            score_best = float(metrics.get(key_best, 0.0))
            score_es = float(metrics.get(key_es, 0.0))

            if score_best > best_val:
                best_val = score_best
                best_epoch_global = epoch_global
                out_path = fold_dir / ckpt_name
                _save_ckpt(
                    out_path=out_path,
                    model=model,
                    cfg=cfg,
                    fold=fold,
                    stage=stage,
                    best_val=best_val,
                    best_epoch_global=best_epoch_global,
                )
                best_ckpt_path = out_path
                print(f"  ↳ saved best model to {out_path} (best {key_best}={best_val:.4f} @E{best_epoch_global})")

            if early_stop and epoch_global >= int(warmup_epochs_global):
                improved = (score_es >= last_best_for_es + float(min_delta))
                if improved:
                    last_best_for_es = score_es
                    patience_left = int(patience)
                else:
                    patience_left -= 1
                    print(
                        f"  ↳ early_stop(S{stage}): no improvement on {key_es} "
                        f"(min_delta={min_delta}). patience_left={patience_left}"
                    )

                if patience_left <= 0:
                    print(
                        f"[fold {fold}] EARLY STOP in stage{stage} at epoch {epoch_global} "
                        f"(best {key_best}={best_val:.4f} @E{best_epoch_global})"
                    )
                    history_rows.append(row)

                    out_path_last = fold_dir / ckpt_name
                    _save_ckpt(
                        out_path=out_path_last,
                        model=model,
                        cfg=cfg,
                        fold=fold,
                        stage=stage,
                        best_val=best_val,
                        best_epoch_global=best_epoch_global,
                    )
                    print(f"  ↳ saved LAST model to {out_path_last} (early stop @E{epoch_global})")
                    return history_rows, best_val, best_epoch_global, epoch_global, best_ckpt_path

        history_rows.append(row)

    end_epoch_global = start_epoch_global + n_epochs
    out_path_last = fold_dir / ckpt_name
    _save_ckpt(
        out_path=out_path_last,
        model=model,
        cfg=cfg,
        fold=fold,
        stage=stage,
        best_val=best_val,
        best_epoch_global=best_epoch_global,
    )
    print(f"  ↳ saved LAST model to {out_path_last} (stage{stage} end @E{end_epoch_global})")

    return history_rows, best_val, best_epoch_global, end_epoch_global, best_ckpt_path


# ----------------------------
# 7) Config
# ----------------------------
@dataclass
class TrainCfg:
    proj_dim: int
    gly_hidden1: int
    gly_hidden2: int
    mesh_hidden1: int
    mesh_hidden2: int
    dropout: float
    normalize_joint: bool
    init_logit_scale: float

    lr_stage1: float
    lr_stage2: float
    weight_decay: float
    epochs_stage1: int
    epochs_stage2: int
    stage2_init: str
    grad_clip: float

    batch_size: int
    num_workers: int
    mesh_batch_size: int

    eval_every: int
    eval_batch_gly: int
    eval_batch_mesh: int
    best_k: int
    best_metric: str
    es_metric: str
    early_stop: bool
    patience_stage1: int
    patience_stage2: int
    min_delta: float
    warmup_epochs: int

    pos_weight_mode: str
    pos_weight_clip_max: float

    seed: int
    device: str
    n_folds: int


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _metric_value_from_cv_summary(summary_df: pd.DataFrame, metric: str) -> float:
    if metric not in summary_df.columns:
        raise KeyError(f"Metric '{metric}' not found in cv_summary columns: {list(summary_df.columns)}")
    vals = summary_df[metric].astype(float).to_numpy()
    if len(vals) == 0:
        return float("-inf")
    return float(np.mean(vals))


# ----------------------------
# 8) CV on DEV
# ----------------------------
def _run_dev_cv5_once(
    *,
    cfg: TrainCfg,
    G: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    dev_pairs: List[Tuple[int, int]],
    out_dir: Path,
    metric: str,
    trial: Optional[Trial] = None,
    row2fold: Dict[int, int],
) -> Tuple[float, pd.DataFrame]:
    set_seed(cfg.seed)
    device = _get_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_ids = list(range(cfg.n_folds))
    gly_in_pairs = sorted({g for (g, _) in dev_pairs})

    eval_topks = [1, 10, 15, 20, 25, 30]
    key_best = cfg.best_metric
    if key_best == "auto":
        key_best = f"mrr@{cfg.best_k}"
    key_es = cfg.es_metric

    for mname in [key_best, key_es, metric]:
        k = _parse_metric_k(mname)
        if k is not None and k not in eval_topks:
            eval_topks.append(k)
    eval_topks = sorted(set(eval_topks))

    all_fold_summaries: List[Dict[str, float]] = []

    for fold in fold_ids:
        train_gly = sorted({g for (g, _) in dev_pairs if row2fold[g] != fold})
        val_gly = sorted({g for (g, _) in dev_pairs if row2fold[g] == fold})
        pairs_val = [(g, m) for (g, m) in dev_pairs if row2fold[g] == fold]
        if not train_gly or not val_gly or not pairs_val:
            continue

        g2true_val = defaultdict(set)
        for g, m in pairs_val:
            g2true_val[g].add(m)

        ds_train = GlycanIndexDataset(train_gly)
        dl = torch.utils.data.DataLoader(
            ds_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model = GlyMeshDualMLP(
            gly_in_dim=G.size(1),
            mesh_in_dim=M.size(1),
            proj_dim=cfg.proj_dim,
            gly_hidden1=cfg.gly_hidden1,
            gly_hidden2=cfg.gly_hidden2,
            mesh_hidden1=cfg.mesh_hidden1,
            mesh_hidden2=cfg.mesh_hidden2,
            dropout=cfg.dropout,
            normalize_joint=cfg.normalize_joint,
            init_logit_scale=cfg.init_logit_scale,
        ).to(device)

        Y_train = Y[train_gly]
        pos_weight = make_pos_weight(
            Y_train=Y_train,
            mode=cfg.pos_weight_mode,
            clip_max=cfg.pos_weight_clip_max,
        )

        fold_history: List[Dict[str, float]] = []

        s1_rows, s1_best, s1_best_epoch, s1_end_epoch, s1_best_path = run_stage(
            stage=1,
            model=model,
            dl=dl,
            G=G,
            M=M,
            Y=Y,
            g2true_val=g2true_val,
            device=device,
            fold=fold,
            cfg=cfg,
            fold_dir=fold_dir,
            start_epoch_global=0,
            n_epochs=cfg.epochs_stage1,
            lr=cfg.lr_stage1,
            weight_decay=cfg.weight_decay,
            pos_weight=pos_weight,
            key_best=key_best,
            key_es=key_es,
            eval_topks=eval_topks,
            early_stop=cfg.early_stop,
            patience=cfg.patience_stage1,
            min_delta=cfg.min_delta,
            warmup_epochs_global=cfg.warmup_epochs,
            ckpt_name="best_dual_mlp_stage1.pth",
        )
        fold_history.extend(s1_rows)

        stage2_start_epoch = s1_end_epoch
        if cfg.stage2_init == "best" and s1_best_path is not None:
            _load_ckpt(s1_best_path, model, device)

        s2_rows, s2_best, s2_best_epoch, s2_end_epoch, s2_best_path = run_stage(
            stage=2,
            model=model,
            dl=dl,
            G=G,
            M=M,
            Y=Y,
            g2true_val=g2true_val,
            device=device,
            fold=fold,
            cfg=cfg,
            fold_dir=fold_dir,
            start_epoch_global=stage2_start_epoch,
            n_epochs=cfg.epochs_stage2,
            lr=cfg.lr_stage2,
            weight_decay=cfg.weight_decay,
            pos_weight=pos_weight,
            key_best=key_best,
            key_es=key_es,
            eval_topks=eval_topks,
            early_stop=cfg.early_stop,
            patience=cfg.patience_stage2,
            min_delta=cfg.min_delta,
            warmup_epochs_global=cfg.warmup_epochs,
            ckpt_name="best_dual_mlp_stage2.pth",
        )
        fold_history.extend(s2_rows)

        fold_best_val = s2_best if (s2_best >= s1_best) else s1_best
        fold_best_epoch = s2_best_epoch if (s2_best >= s1_best) else s1_best_epoch
        fold_best_stage = 2 if (s2_best >= s1_best) else 1

        pd.DataFrame(fold_history).to_csv(fold_dir / "history.csv", index=False)

        all_fold_summaries.append({
            "fold": float(fold),
            "best_stage": float(fold_best_stage),
            "best_epoch": float(fold_best_epoch),
            "best_metric": str(key_best),
            key_best: float(fold_best_val),
            "early_stop_metric": str(key_es),
            "n_train_glycans": float(len(train_gly)),
            "n_val_glycans": float(len(val_gly)),
            "n_val_pairs": float(len(pairs_val)),
            "stage1_best_epoch": float(s1_best_epoch),
            "stage1_best": float(s1_best),
            "stage2_best_epoch": float(s2_best_epoch),
            "stage2_best": float(s2_best),
            "stage2_init": str(cfg.stage2_init),
            "n_unique_glycans_in_pairs": float(len(gly_in_pairs)),
        })

        if trial is not None:
            tmp_df = pd.DataFrame(all_fold_summaries)
            current = _metric_value_from_cv_summary(tmp_df, metric)
            trial.report(current, step=len(all_fold_summaries))
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at fold {fold} with interim {metric}={current:.6f}")

    summary_df = pd.DataFrame(all_fold_summaries)
    summary_df.to_csv(out_dir / "cv_summary.csv", index=False)

    obj = _metric_value_from_cv_summary(summary_df, metric)
    return obj, summary_df


# ----------------------------
# 9) Final train on ALL DEV
# ----------------------------
@torch.no_grad()
def _eval_on_test(
    *,
    model: GlyMeshDualMLP,
    G: torch.Tensor,
    M: torch.Tensor,
    test_pairs: List[Tuple[int, int]],
    device: torch.device,
    topks: List[int],
    eval_batch_gly: int,
    eval_batch_mesh: int,
) -> Dict[str, float]:
    g2true_test = defaultdict(set)
    for g, m in test_pairs:
        g2true_test[g].add(m)

    metrics = eval_dual_ranking(
        model=model,
        G=G,
        M=M,
        g2true=g2true_test,
        device=device,
        batch_size_gly=eval_batch_gly,
        batch_size_mesh=eval_batch_mesh,
        topks=topks,
    )
    return {k: float(v) for k, v in metrics.items()}


def _train_two_stage_no_val(
    *,
    cfg: TrainCfg,
    G: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    dev_pairs: List[Tuple[int, int]],
    out_dir: Path,
    device: torch.device,
) -> GlyMeshDualMLP:
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_gly = sorted({g for (g, _) in dev_pairs})
    ds_train = GlycanIndexDataset(dev_gly)
    dl = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GlyMeshDualMLP(
        gly_in_dim=G.size(1),
        mesh_in_dim=M.size(1),
        proj_dim=cfg.proj_dim,
        gly_hidden1=cfg.gly_hidden1,
        gly_hidden2=cfg.gly_hidden2,
        mesh_hidden1=cfg.mesh_hidden1,
        mesh_hidden2=cfg.mesh_hidden2,
        dropout=cfg.dropout,
        normalize_joint=cfg.normalize_joint,
        init_logit_scale=cfg.init_logit_scale,
    ).to(device)

    Y_train = Y[dev_gly]
    pos_weight = make_pos_weight(
        Y_train=Y_train,
        mode=cfg.pos_weight_mode,
        clip_max=cfg.pos_weight_clip_max,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=(pos_weight.to(device) if pos_weight is not None else None)
    )

    def train_stage(lr: float, n_epochs: int, stage_name: str) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(cfg.weight_decay))
        for ep in range(1, n_epochs + 1):
            model.train()
            loss_ema = 0.0
            for g_idx in dl:
                x_g = G[g_idx].to(device)
                y = Y[g_idx].to(device)

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

                loss_ema = loss_ema * 0.9 + float(loss.item()) * 0.1

            print(f"[final_dev_train] {stage_name} epoch={ep}/{n_epochs} train_loss_ema={loss_ema:.6f}")

    train_stage(cfg.lr_stage1, cfg.epochs_stage1, "stage1")
    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__},
        out_dir / "final_ckpt_stage1.pth",
    )

    train_stage(cfg.lr_stage2, cfg.epochs_stage2, "stage2")
    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__},
        out_dir / "final_ckpt_stage2.pth",
    )

    return model


# ----------------------------
# 10) Optuna search space
# ----------------------------
def suggest_params(trial: Trial) -> Dict[str, Any]:
    proj_dim = trial.suggest_categorical("proj_dim", [256, 512, 768])

    gly_hidden1 = trial.suggest_categorical("gly_hidden1", [512, 1024, 1536])
    gly_hidden2 = trial.suggest_categorical("gly_hidden2", [0, 256, 512])
    mesh_hidden1 = trial.suggest_categorical("mesh_hidden1", [512, 1024, 1536])
    mesh_hidden2 = trial.suggest_categorical("mesh_hidden2", [0, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    normalize_joint = trial.suggest_categorical("normalize_joint", [True, False])
    init_logit_scale = trial.suggest_float("init_logit_scale", 0.5, 20.0, log=True)

    lr_stage1 = trial.suggest_float("lr_stage1", 1e-4, 5e-3, log=True)
    lr_stage2 = trial.suggest_float("lr_stage2", 5e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    mesh_batch_size = trial.suggest_categorical("mesh_batch_size", [1024, 2048, 4096])

    pos_weight_mode = trial.suggest_categorical("pos_weight_mode", ["none", "balanced"])
    pos_weight_clip_max = trial.suggest_categorical("pos_weight_clip_max", [20.0, 50.0, 100.0])

    return dict(
        proj_dim=int(proj_dim),
        gly_hidden1=int(gly_hidden1),
        gly_hidden2=int(gly_hidden2),
        mesh_hidden1=int(mesh_hidden1),
        mesh_hidden2=int(mesh_hidden2),
        dropout=float(dropout),
        normalize_joint=bool(normalize_joint),
        init_logit_scale=float(init_logit_scale),
        lr_stage1=float(lr_stage1),
        lr_stage2=float(lr_stage2),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        mesh_batch_size=int(mesh_batch_size),
        pos_weight_mode=str(pos_weight_mode),
        pos_weight_clip_max=float(pos_weight_clip_max),
    )


def objective_factory(
    args,
    G: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    dev_pairs: List[Tuple[int, int]],
    gly_ids: List[str],
    mesh_ids: List[str],
    row2fold_dev: Dict[int, int],
):
    metric = args.metric

    def objective(trial: Trial) -> float:
        trial_seed = (args.seed * 1000003 + trial.number) & 0xFFFFFFFF
        hp = suggest_params(trial)

        best_metric = args.best_metric
        if best_metric == "auto":
            best_metric = metric

        cfg = TrainCfg(
            proj_dim=int(hp["proj_dim"]),
            gly_hidden1=int(hp["gly_hidden1"]),
            gly_hidden2=int(hp["gly_hidden2"]),
            mesh_hidden1=int(hp["mesh_hidden1"]),
            mesh_hidden2=int(hp["mesh_hidden2"]),
            dropout=float(hp["dropout"]),
            normalize_joint=bool(hp["normalize_joint"]),
            init_logit_scale=float(hp["init_logit_scale"]),

            lr_stage1=float(hp["lr_stage1"]),
            lr_stage2=float(hp["lr_stage2"]),
            weight_decay=float(hp["weight_decay"]),
            epochs_stage1=int(args.epochs_stage1),
            epochs_stage2=int(args.epochs_stage2),
            stage2_init=str(args.stage2_init),
            grad_clip=float(args.grad_clip),

            batch_size=int(hp["batch_size"]),
            num_workers=int(args.num_workers),
            mesh_batch_size=int(hp["mesh_batch_size"]),

            eval_every=int(args.eval_every),
            eval_batch_gly=int(args.eval_batch_gly),
            eval_batch_mesh=int(args.eval_batch_mesh),
            best_k=int(args.best_k),
            best_metric=str(best_metric),
            es_metric=str(args.es_metric),
            early_stop=bool(args.early_stop),
            patience_stage1=int(args.patience_stage1),
            patience_stage2=int(args.patience_stage2),
            min_delta=float(args.min_delta),
            warmup_epochs=int(args.warmup_epochs),

            pos_weight_mode=str(hp["pos_weight_mode"]),
            pos_weight_clip_max=float(hp["pos_weight_clip_max"]),

            seed=int(trial_seed),
            device=str(args.device),
            n_folds=int(args.n_folds),
        )

        trial_dir = Path(args.study_dir) / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        params_out = {
            "model": "glycan_mesh_dual_mlp_multilabel",
            "trial_number": int(trial.number),
            "trial_seed": int(trial_seed),
            "objective_metric": str(metric),
            "params": _coerce_jsonable(hp),
            "fixed": {
                "epochs_stage1": int(args.epochs_stage1),
                "epochs_stage2": int(args.epochs_stage2),
                "n_folds": int(args.n_folds),
                "test_frac": float(args.test_frac),
                "test_seed": int(args.test_seed),
                "n_mesh_labels": int(len(mesh_ids)),
            },
            "mesh_emb_csv": str(args.mesh_emb_csv),
            "n_mesh_labels": int(len(mesh_ids)),
        }
        (trial_dir / "params.json").write_text(json.dumps(params_out, indent=2), encoding="utf-8")

        val, _ = _run_dev_cv5_once(
            cfg=cfg,
            G=G,
            M=M,
            Y=Y,
            dev_pairs=dev_pairs,
            out_dir=trial_dir,
            metric=metric,
            trial=trial if args.prune else None,
            row2fold=row2fold_dev,
        )

        trial.set_user_attr("dev_cv_mean_" + metric, float(val))
        return float(val)

    return objective


# ----------------------------
# 11) CLI
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

    # fixed test split
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--test_seed", type=int, default=123)

    # study
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--study_name", default="glycan_mesh_dual_mlp_holdout_test_cv5")
    ap.add_argument("--storage", default="")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--timeout_min", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_seed", type=int, default=42)

    # metric
    ap.add_argument("--metric", default="recall@25")
    ap.add_argument("--best_metric", default="auto")
    ap.add_argument("--best_k", type=int, default=25)

    # schedule
    ap.add_argument("--epochs_stage1", type=int, default=50)
    ap.add_argument("--epochs_stage2", type=int, default=100)
    ap.add_argument("--stage2_init", type=str, default="best", choices=["best", "last"])

    # early stopping
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--es_metric", default="recall@25")
    ap.add_argument("--patience_stage1", type=int, default=5)
    ap.add_argument("--patience_stage2", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    # batch / eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch_gly", type=int, default=512)
    ap.add_argument("--eval_batch_mesh", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # device / cv
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_folds", type=int, default=5)

    # pruning
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--pruner", default="median", choices=["median", "nop"])
    ap.add_argument("--startup_trials", type=int, default=5)

    return ap


def main():
    args = build_argparser().parse_args()

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(parents=True, exist_ok=True)

    storage = args.storage.strip() if args.storage.strip() else f"sqlite:///{(study_dir / 'study.db').as_posix()}"

    G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row = load_embeddings_from_csv(
        args.gly_emb_csv,
        args.mesh_emb_csv,
        normalize_gly=(not args.no_l2_norm_gly),
        normalize_mesh=(not args.no_l2_norm_mesh),
    )

    pairs, missing_gly, missing_mesh = build_pairs_from_csv(
        pairs_csv=args.pairs_csv,
        gly_id2row=gly_id2row,
        mesh_id2row=mesh_id2row,
        gly_id_col=args.gly_id_col,
        mesh_list_col=args.mesh_list_col,
        mesh_sep=args.mesh_sep,
    )
    if not pairs:
        raise ValueError("No valid pairs after ID mapping.")

    print(
        f"[pairs] built={len(pairs)} "
        f"missing_gly={missing_gly} "
        f"missing_mesh={missing_mesh} "
        f"n_mesh_labels(full_candidate)={len(mesh_ids)}"
    )

    Y, _ = build_multilabel_targets(
        pairs=pairs,
        n_gly=G.size(0),
        n_mesh=len(mesh_ids),
    )

    test_glycans, dev_glycans = make_fixed_test_split(
        pairs=pairs,
        test_frac=float(args.test_frac),
        seed=int(args.test_seed),
    )
    dev_pairs = filter_pairs_by_glycans(pairs, dev_glycans)
    test_pairs = filter_pairs_by_glycans(pairs, test_glycans)

    if not dev_pairs or not test_pairs:
        raise RuntimeError(f"Split produced empty DEV or TEST. dev_pairs={len(dev_pairs)} test_pairs={len(test_pairs)}")

    cv_seed = int(args.cv_seed)
    row2fold_dev, gly_in_dev_pairs = make_kfold_assignment_on_unique_glycans(
        pairs=dev_pairs,
        n_folds=int(args.n_folds),
        seed=cv_seed,
    )
    dev_gly_set = {g for (g, _) in dev_pairs}
    if any(g not in row2fold_dev for g in dev_gly_set):
        raise RuntimeError("row2fold_dev missing some glycans in dev_pairs.")

    assign_df = pd.DataFrame({
        "gly_row": [int(g) for g in gly_in_dev_pairs],
        "glytoucan_ac": [str(gly_ids[int(g)]) for g in gly_in_dev_pairs],
        "fold": [int(row2fold_dev[int(g)]) for g in gly_in_dev_pairs],
    }).sort_values(["fold", "gly_row"]).reset_index(drop=True)

    assign_path = study_dir / "fold_assignment_dev.csv"
    assign_df.to_csv(assign_path, index=False)
    print(f"[split] wrote DEV fold assignment: {assign_path.as_posix()} (cv_seed={cv_seed})")

    split_out = {
        "test_frac": float(args.test_frac),
        "test_seed": int(args.test_seed),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_dev": int(len(dev_pairs)),
        "n_pairs_test": int(len(test_pairs)),
        "n_unique_glycans_total": int(len({g for (g, _) in pairs})),
        "n_unique_glycans_dev": int(len(dev_glycans)),
        "n_unique_glycans_test": int(len(test_glycans)),
        "n_mesh_labels": int(len(mesh_ids)),
        "test_glycan_rows": sorted(int(x) for x in test_glycans),
    }
    (study_dir / "split_test_glycans.json").write_text(json.dumps(split_out, indent=2), encoding="utf-8")
    print(
        f"[split] DEV pairs={len(dev_pairs)} TEST pairs={len(test_pairs)} "
        f"DEV glycans={len(dev_glycans)} TEST glycans={len(test_glycans)}"
    )

    if args.prune:
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=int(args.startup_trials))
            if args.pruner == "median"
            else optuna.pruners.NopPruner()
        )
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = objective_factory(args, G, M, Y, dev_pairs, gly_ids, mesh_ids, row2fold_dev)

    timeout = None if int(args.timeout_min) <= 0 else int(args.timeout_min) * 60

    t0 = time.time()
    study.optimize(objective, n_trials=int(args.n_trials), timeout=timeout, gc_after_trial=True)
    t1 = time.time()

    print(f"[optuna done] elapsed_sec={t1 - t0:.1f}")
    print("Best value (DEV CV mean):", study.best_value)
    print("Best params:", study.best_params)

    best_out = {
        "study_name": args.study_name,
        "storage": storage,
        "objective_metric": args.metric,
        "best_value_dev_cv_mean": float(study.best_value),
        "best_params": _coerce_jsonable(dict(study.best_params)),
        "model": "glycan_mesh_dual_mlp_multilabel",
        "n_mesh_labels": int(len(mesh_ids)),
        "test_split": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
        },
        "mesh_emb_csv": str(args.mesh_emb_csv),
    }
    (study_dir / "study_best.json").write_text(json.dumps(best_out, indent=2), encoding="utf-8")

    print("\n[final] Retrain on ALL DEV with best HP (no val), then evaluate on fixed TEST.")

    bp = dict(study.best_params)
    final_cfg = TrainCfg(
        proj_dim=int(bp["proj_dim"]),
        gly_hidden1=int(bp["gly_hidden1"]),
        gly_hidden2=int(bp["gly_hidden2"]),
        mesh_hidden1=int(bp["mesh_hidden1"]),
        mesh_hidden2=int(bp["mesh_hidden2"]),
        dropout=float(bp["dropout"]),
        normalize_joint=bool(bp["normalize_joint"]),
        init_logit_scale=float(bp["init_logit_scale"]),

        lr_stage1=float(bp["lr_stage1"]),
        lr_stage2=float(bp["lr_stage2"]),
        weight_decay=float(bp["weight_decay"]),
        epochs_stage1=int(args.epochs_stage1),
        epochs_stage2=int(args.epochs_stage2),
        stage2_init=str(args.stage2_init),
        grad_clip=float(args.grad_clip),

        batch_size=int(bp["batch_size"]),
        num_workers=int(args.num_workers),
        mesh_batch_size=int(bp["mesh_batch_size"]),

        eval_every=int(args.eval_every),
        eval_batch_gly=int(args.eval_batch_gly),
        eval_batch_mesh=int(args.eval_batch_mesh),
        best_k=int(args.best_k),
        best_metric=str(args.best_metric if args.best_metric != "auto" else args.metric),
        es_metric=str(args.es_metric),
        early_stop=False,
        patience_stage1=int(args.patience_stage1),
        patience_stage2=int(args.patience_stage2),
        min_delta=float(args.min_delta),
        warmup_epochs=int(args.warmup_epochs),

        pos_weight_mode=str(bp["pos_weight_mode"]),
        pos_weight_clip_max=float(bp["pos_weight_clip_max"]),

        seed=int(args.seed),
        device=str(args.device),
        n_folds=int(args.n_folds),
    )

    device = _get_device(final_cfg.device)
    final_dir = study_dir / "final_dev_train"
    model = _train_two_stage_no_val(
        cfg=final_cfg,
        G=G,
        M=M,
        Y=Y,
        dev_pairs=dev_pairs,
        out_dir=final_dir,
        device=device,
    )

    eval_topks = [1, 10, 15, 20, 25, 30]
    for mname in [args.metric]:
        k = _parse_metric_k(mname)
        if k is not None and k not in eval_topks:
            eval_topks.append(k)
    eval_topks = sorted(set(eval_topks))

    test_metrics = _eval_on_test(
        model=model,
        G=G,
        M=M,
        test_pairs=test_pairs,
        device=device,
        topks=eval_topks,
        eval_batch_gly=int(args.eval_batch_gly),
        eval_batch_mesh=int(args.eval_batch_mesh),
    )

    test_out = {
        "model": "glycan_mesh_dual_mlp_multilabel",
        "objective_metric": str(args.metric),
        "best_params": _coerce_jsonable(dict(study.best_params)),
        "final_dev_train": {
            "epochs_stage1": int(args.epochs_stage1),
            "epochs_stage2": int(args.epochs_stage2),
            "batch_size": int(final_cfg.batch_size),
            "mesh_batch_size": int(final_cfg.mesh_batch_size),
            "seed": int(final_cfg.seed),
            "n_mesh_labels": int(len(mesh_ids)),
        },
        "fixed_test": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
            "n_pairs_test": int(len(test_pairs)),
            "n_unique_glycans_test": int(len(test_glycans)),
        },
        "test_metrics": test_metrics,
        "mesh_emb_csv": str(args.mesh_emb_csv),
    }
    (final_dir / "test_metrics.json").write_text(json.dumps(test_out, indent=2), encoding="utf-8")

    print("[final] wrote:", (final_dir / "test_metrics.json").as_posix())
    print("[final] TEST metric summary:")
    keys_show = [f"hit@{k}" for k in eval_topks] + [f"recall@{k}" for k in eval_topks] + [f"mrr@{k}" for k in eval_topks]
    keys_show = [k for k in keys_show if k in test_metrics]
    for k in keys_show:
        print(f"  {k}: {test_metrics[k]:.6f}")

    print("\n[done] Glycan+MeSH dual-MLP multi-label baseline complete.")


if __name__ == "__main__":
    main()


# Example (uses a NEW output directory so prior runs are not overwritten)
# python optuna_glycan_mesh_dual_mlp_holdout_test_cv5.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --study_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/optuna_holdout_cv5_run01 \
#   --study_name glycan_mesh_dual_mlp_holdout_test_cv5_run01 \
#   --n_trials 100 \
#   --metric recall@25 \
#   --test_frac 0.15 \
#   --early_stop \
#   --prune \
#   --device cuda

# python optuna_glycan_mesh_dual_mlp_holdout_test_cv5.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/biobert_description_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --study_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/optuna_holdout_cv5_biobert \
#   --study_name glycan_mesh_dual_mlp_holdout_test_cv5_biobert \
#   --n_trials 100 \
#   --metric recall@25 \
#   --test_frac 0.15 \
#   --early_stop \
#   --prune \
#   --device cuda

# python optuna_glycan_mesh_dual_mlp_holdout_test_cv5.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/pubmedbert_description_meanpool_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --study_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/optuna_holdout_cv5_pubmedbert \
#   --study_name glycan_mesh_dual_mlp_holdout_test_cv5_pubmedbert \
#   --n_trials 100 \
#   --metric recall@25 \
#   --test_frac 0.15 \
#   --early_stop \
#   --prune \
#   --device cuda

# python optuna_glycan_mesh_dual_mlp_holdout_test_cv5.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/medcpt_description_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --study_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/optuna_holdout_cv5_medcpt \
#   --study_name glycan_mesh_dual_mlp_holdout_test_cv5_medcpt \
#   --n_trials 100 \
#   --metric recall@25 \
#   --test_frac 0.15 \
#   --early_stop \
#   --prune \
#   --device cuda