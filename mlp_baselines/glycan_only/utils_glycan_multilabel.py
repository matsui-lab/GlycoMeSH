#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils_glycan_multilabel.py
==========================

Components for glycan-only multi-label classification:
- seed / normalization
- MLP multi-label classifier
- data IO helpers
- label matrix construction
- ranking-based evaluation (glycan -> MeSH labels)
- training helpers (one epoch, stage runner, ckpt IO)
- fold assignment
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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


# ----------------------------
# 2) Model
# ----------------------------
class GlycanMLPClassifier(nn.Module):
    """
    Multi-label classifier:
      glycan embedding -> MLP -> logits over all MeSH labels
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden1: int = 1024,
        hidden2: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

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
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits


# ----------------------------
# 3) Dataset
# ----------------------------
class GlycanMultiLabelDataset(Dataset):
    """
    Dataset of glycan row indices.
    Targets are taken from a dense label matrix Y[g_idx, :]
    """

    def __init__(self, gly_indices: List[int]):
        self.gly_indices = sorted(set(int(x) for x in gly_indices))
        if len(self.gly_indices) == 0:
            raise ValueError("Empty glycan index list.")

    def __len__(self) -> int:
        return len(self.gly_indices)

    def __getitem__(self, i: int) -> int:
        return self.gly_indices[i]


# ----------------------------
# 4) Metric parsing
# ----------------------------
def _parse_metric_k(metric: str) -> Optional[int]:
    """
    Supports: hit@10, recall@20, precision@30, mrr@25
    """
    m = re.match(r"^(hit|recall|precision|mrr)@(\d+)$", metric.strip())
    if not m:
        return None
    return int(m.group(2))


# ----------------------------
# 5) Data IO helpers
# ----------------------------
def load_gly_embeddings_from_csv(
    gly_emb_csv: str,
    normalize: bool = True,
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    """
    Returns:
      G: (N_g, D) torch.Tensor on CPU
      gly_ids: list[str]
      gly_id2row: Dict[str, int]
    """
    gly_df = pd.read_csv(gly_emb_csv)
    gly_id_col = gly_df.columns[0]

    gly_ids = gly_df[gly_id_col].astype(str).tolist()
    gly_id2row = {gid: i for i, gid in enumerate(gly_ids)}

    G_np = gly_df.drop(columns=[gly_id_col]).to_numpy(dtype=np.float32)
    G = torch.from_numpy(G_np)
    if normalize:
        G = l2_normalize(G)

    return G, gly_ids, gly_id2row

def load_mesh_ids_from_embedding_csv(
    mesh_emb_csv: str,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Returns:
      mesh_ids: list[str]
      mesh_id2row: Dict[str, int]

    Assumes the first column of mesh_emb_csv is the MeSH ID column.
    """
    mesh_df = pd.read_csv(mesh_emb_csv)
    mesh_id_col = mesh_df.columns[0]

    mesh_ids = mesh_df[mesh_id_col].astype(str).tolist()
    mesh_id2row = {mid: i for i, mid in enumerate(mesh_ids)}
    return mesh_ids, mesh_id2row

def build_pairs_from_csv_with_given_mesh_vocab(
    pairs_csv: str,
    gly_id2row: Dict[str, int],
    mesh_id2row: Dict[str, int],
    gly_id_col: str = "glytoucan_ac",
    mesh_list_col: str = "descriptor_ui_list",
    mesh_sep: str = ";",
) -> Tuple[List[Tuple[int, int]], int, int]:
    """
    Map pairs_csv onto a pre-defined full MeSH vocabulary.

    Returns:
      pairs: List[(gly_row, mesh_row)]
      missing_gly: number of glycans in pairs_csv absent from gly embeddings
      missing_mesh: number of mesh labels in pairs_csv absent from mesh vocabulary
    """
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
    """
    Returns:
      Y: (N_g, N_mesh) multi-hot target matrix
      g2true: gly_idx -> set(mesh_idx)
    """
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
    """
    Returns:
      row2fold: glycan_row_index -> fold_id
      gly_in_pairs: sorted unique glycan indices
    """
    gly_in_pairs = sorted({g for (g, _) in pairs})
    if len(gly_in_pairs) < n_folds:
        raise ValueError(
            f"Not enough unique glycans for n_folds={n_folds}. "
            f"unique_glycans_in_pairs={len(gly_in_pairs)}"
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    row2fold: Dict[int, int] = {}
    for fold_id, (_, val_idx) in enumerate(kf.split(gly_in_pairs)):
        for j in val_idx:
            g_row = gly_in_pairs[j]
            row2fold[g_row] = int(fold_id)

    if len(row2fold) != len(gly_in_pairs):
        raise RuntimeError("KFold assignment failed: some glycans did not receive a fold id.")

    return row2fold, gly_in_pairs


# ----------------------------
# 6) Fixed holdout split
# ----------------------------
def make_fixed_test_split(
    *,
    pairs: List[Tuple[int, int]],
    test_frac: float,
    seed: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Returns:
      test_glycans_set, dev_glycans_set
    """
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
# 7) Evaluation
# ----------------------------
@torch.no_grad()
def eval_multilabel_ranking(
    model: nn.Module,
    G: torch.Tensor,
    g2true: Dict[int, Set[int]],
    device: torch.device,
    batch_size: int = 1024,
    topks: List[int] | Tuple[int, ...] = (1, 10, 15, 20, 25, 30),
) -> Dict[str, float]:
    model.eval()

    eval_gly = sorted(int(g) for g in g2true.keys())
    if len(eval_gly) == 0:
        return {"n_eval": 0.0}

    topks = sorted(set(int(k) for k in topks))
    max_k = max(topks)

    logits_map: Dict[int, torch.Tensor] = {}

    for s in range(0, len(eval_gly), batch_size):
        idx = eval_gly[s:s + batch_size]
        x = G[idx].to(device)
        logits = model(x).cpu()   # shape: [B, n_mesh]
        for j, g_idx in enumerate(idx):
            logits_map[int(g_idx)] = logits[j]

    hit = {k: 0 for k in topks}
    rec = {k: 0.0 for k in topks}
    prec = {k: 0.0 for k in topks}
    mrr = {k: 0.0 for k in topks}
    n = 0

    for g_idx, true_set in g2true.items():
        scores = logits_map[int(g_idx)]
        top_all = torch.topk(scores, k=max_k).indices.tolist()

        true_set = set(true_set)
        denom_true = max(1, len(true_set))
        n += 1

        for k in topks:
            topk = top_all[:k]

            if any(m in true_set for m in topk):
                hit[k] += 1

            inter = len(set(topk).intersection(true_set))
            rec[k] += inter / denom_true13
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
# 8) Training helpers
# ----------------------------
def make_pos_weight(
    Y_train: torch.Tensor,
    mode: str = "none",
    clip_min: float = 1.0,
    clip_max: float = 100.0,
) -> Optional[torch.Tensor]:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    mode:
      - "none": return None
      - "balanced": neg_count / pos_count per label (clipped)
    """
    mode = str(mode).lower()
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown pos_weight mode: {mode}")

    pos = Y_train.sum(dim=0)                           # (C,)
    total = torch.tensor(float(Y_train.size(0)))
    neg = total - pos
    pw = neg / torch.clamp(pos, min=1.0)
    pw = torch.clamp(pw, min=float(clip_min), max=float(clip_max))
    return pw.to(dtype=torch.float32)


def _train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    G: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    desc: str,
) -> float:
    model.train()
    loss_ema = 0.0

    pbar = tqdm(dl, desc=desc, leave=True)
    for g_idx in pbar:
        x = G[g_idx].to(device)
        y = Y[g_idx].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_ema = loss_ema * 0.9 + float(loss.item()) * 0.1
        pbar.set_postfix({"loss": f"{loss_ema:.4f}"})

    return float(loss_ema)


def _save_ckpt(
    out_path: Path,
    model: nn.Module,
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


def _load_ckpt(
    ckpt_path: Path,
    model: nn.Module,
    device: torch.device,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def run_stage(
    *,
    stage: int,
    model: nn.Module,
    dl: DataLoader,
    G: torch.Tensor,
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
    """
    Returns:
      history_rows, best_val(key_best), best_epoch_global, end_epoch_global, best_ckpt_path
    """
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
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
            Y=Y,
            device=device,
            opt=opt,
            criterion=criterion,
            grad_clip=float(cfg.grad_clip),
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
            metrics = eval_multilabel_ranking(
                model=model,
                G=G,
                g2true=g2true_val,
                device=device,
                batch_size=int(cfg.eval_batch),
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
                    print(f"  ↳ early_stop(S{stage}): no improvement on {key_es} (min_delta={min_delta}). patience_left={patience_left}")

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


__all__ = [
    "set_seed",
    "l2_normalize",
    "GlycanMLPClassifier",
    "GlycanMultiLabelDataset",
    "_parse_metric_k",
    "load_gly_embeddings_from_csv",
    "load_mesh_ids_from_embedding_csv","build_pairs_from_csv_with_given_mesh_vocab",
    "build_multilabel_targets",
    "make_kfold_assignment_on_unique_glycans",
    "make_fixed_test_split",
    "filter_pairs_by_glycans",
    "eval_multilabel_ranking",
    "make_pos_weight",
    "_train_one_epoch",
    "_save_ckpt",
    "_load_ckpt",
    "run_stage",
]