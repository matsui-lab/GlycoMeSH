#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py
========
Reusable components for cross-modal multi-positive contrastive learning:
- seed / normalization
- ProjectionHead
- Dataset + collate for multi-positive batching
- StableRep-style multi-positive loss
- retrieval evaluation (glycan -> mesh)
- training helpers (one epoch, stage runner, ckpt IO)
- data IO helpers (embeddings, pairs, kfold assignment)

Intended to be imported by:
- train_cross_modal_stablerep_normal_cv.py
- Optuna objective scripts (same directory)
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.model_selection import KFold


# ----------------------------
# 1) Reproducibility utilities
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


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix, assuming inputs are L2-normalized."""
    return a @ b.T


# ----------------------------
# 2) Projection heads
# ----------------------------
class ProjectionHead(nn.Module):
    """MLP projection head with L2-normalized output."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden is None:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


# ----------------------------
# 3) Dataset for multi-positive batching
# ----------------------------
class GlyMeshPairsDataset(Dataset):
    """
    Stores mapping glycan_idx -> list of mesh_idx so that a collate_fn can sample
    multiple positives per glycan anchor.
    """

    def __init__(self, pairs: List[Tuple[int, int]]):
        self.g2pairs: Dict[int, List[int]] = defaultdict(list)
        for g, m in pairs:
            self.g2pairs[g].append(m)

        # dedup
        for g in list(self.g2pairs.keys()):
            self.g2pairs[g] = sorted(set(self.g2pairs[g]))

        self.gly_indices = sorted(self.g2pairs.keys())
        if not self.gly_indices:
            raise ValueError("No labeled glycans found after filtering pairs.")

    def __len__(self) -> int:
        return len(self.gly_indices)

    def __getitem__(self, i: int) -> int:
        return self.gly_indices[i]


@dataclass
class BatchConfig:
    batch_glycans: int = 128
    pos_per_glycan: int = 2
    allow_pos_replacement: bool = True


def make_collate_fn(ds: GlyMeshPairsDataset, cfg: BatchConfig):
    """
    Collate returns a batch of B = sum_i P_i pair samples:
      g_idx:   (B,)
      m_idx:   (B,)
      group:   (B,)  group id for each glycan anchor (0..n-1)
    """
    g2pairs = ds.g2pairs

    def collate(gly_idxs: List[int]):
        if len(gly_idxs) > cfg.batch_glycans:
            gly_idxs = gly_idxs[: cfg.batch_glycans]

        g_list: List[int] = []
        m_list: List[int] = []
        group_list: List[int] = []

        for gi, g in enumerate(gly_idxs):
            pairs = g2pairs[g]
            if not pairs:
                continue

            P = min(cfg.pos_per_glycan, len(pairs))
            if len(pairs) >= P:
                chosen = random.sample(pairs, P)
            else:
                if cfg.allow_pos_replacement:
                    chosen = [random.choice(pairs) for _ in range(P)]
                else:
                    chosen = pairs

            for m in chosen:
                g_list.append(g)
                m_list.append(m)
                group_list.append(gi)

        return (
            torch.tensor(g_list, dtype=torch.long),
            torch.tensor(m_list, dtype=torch.long),
            torch.tensor(group_list, dtype=torch.long),
        )

    return collate


# ----------------------------
# 4) StableRep-style multi-positive loss
# ----------------------------
def build_pos_mask(group: torch.Tensor) -> torch.Tensor:
    """pos_mask[i,j]=1 if candidate j is positive for anchor i (same glycan group)."""
    return (group[:, None] == group[None, :]).float()


def build_p_distribution(pos_mask: torch.Tensor, self_mask: bool = True) -> torch.Tensor:
    """
    p_{i,j} = 1/|P_i| if j in P_i else 0. Optionally removes diagonal.
    """
    p = pos_mask.clone()
    if self_mask:
        p.fill_diagonal_(0.0)

    row_sum = p.sum(dim=1, keepdim=True)
    p = torch.where(row_sum > 0, p / row_sum, p)
    return p


def stablerep_loss(
    z_g: torch.Tensor,     # (B,d) glycan projected (L2 normalized)
    z_m: torch.Tensor,     # (B,d) mesh projected   (L2 normalized)
    group: torch.Tensor,   # (B,) glycan group ids
    tau: float = 0.2,
    self_mask: bool = True,
) -> torch.Tensor:
    """
    L = H(p, q) with:
      logits = (z_g z_m^T)/tau
      q = softmax(logits)   (optionally masking diagonal)
      p = uniform over positives per row (optionally masking diagonal)
    """
    B = z_g.size(0)
    device = z_g.device

    logits = cosine_sim(z_g, z_m) / tau  # (B,B)

    pos_mask = build_pos_mask(group)  # includes diagonal
    p = build_p_distribution(pos_mask, self_mask=self_mask)
    valid = (p.sum(dim=1) > 0)

    if self_mask:
        diag = torch.arange(B, device=device)
        logits[diag, diag] = -1e9

    log_q = F.log_softmax(logits, dim=1)
    loss_per_row = -(p * log_q).sum(dim=1)

    return loss_per_row[valid].mean()


# ----------------------------
# 5) Retrieval evaluation
# ----------------------------
@torch.no_grad()
def eval_retrieval_gly_to_mesh(
    proj_g: nn.Module,
    proj_m: nn.Module,
    G: torch.Tensor,                      # (N_g, Dg)
    M: torch.Tensor,                      # (N_m, Dm)
    g2true: Dict[int, set],               # gly_idx -> set(mesh_idx)
    device: torch.device,
    batch_size: int = 1024,
    topks: List[int] | Tuple[int, ...] = (1, 10, 15, 20, 25, 30),
) -> Dict[str, float]:
    proj_g.eval()
    proj_m.eval()

    topks = sorted(set(int(k) for k in topks))
    max_k = max(topks)

    # project all glycans
    zg_all = []
    for s in range(0, G.size(0), batch_size):
        e = min(G.size(0), s + batch_size)
        zg_all.append(proj_g(G[s:e].to(device)).cpu())
    zg_all = torch.cat(zg_all, dim=0)

    # project all mesh
    zm_all = []
    for s in range(0, M.size(0), batch_size):
        e = min(M.size(0), s + batch_size)
        zm_all.append(proj_m(M[s:e].to(device)).cpu())
    zm_all = torch.cat(zm_all, dim=0)

    zmT = zm_all.t().contiguous()

    hit = {k: 0 for k in topks}
    rec = {k: 0.0 for k in topks}
    prec = {k: 0.0 for k in topks}
    mrr = {k: 0.0 for k in topks}
    n = 0

    for g_idx, true_set in g2true.items():
        scores = (zg_all[g_idx: g_idx + 1] @ zmT).squeeze(0)
        top_all = torch.topk(scores, k=max_k).indices.tolist()
        n += 1

        true_set = set(true_set)
        denom_true = max(1, len(true_set))

        for k in topks:
            topk = top_all[:k]

            # hit@k
            if any(m in true_set for m in topk):
                hit[k] += 1

            inter = len(set(topk).intersection(true_set))

            # recall@k, precision@k
            rec[k] += inter / denom_true
            prec[k] += inter / float(k)

            # mrr@k
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
# 6) Helpers: training one epoch, and stage runner
# ----------------------------
def _train_one_epoch(
    proj_g: nn.Module,
    proj_m: nn.Module,
    dl: DataLoader,
    G: torch.Tensor,
    M: torch.Tensor,
    device: torch.device,
    tau: float,
    grad_clip: float,
    opt: torch.optim.Optimizer,
    desc: str,
) -> float:
    proj_g.train()
    proj_m.train()
    loss_ema = 0.0

    pbar = tqdm(dl, desc=desc, leave=True)
    for g_idx, m_idx, group in pbar:
        g_emb = G[g_idx].to(device)
        m_emb = M[m_idx].to(device)

        zg = proj_g(g_emb)
        zm = proj_m(m_emb)

        loss = stablerep_loss(
            z_g=zg,
            z_m=zm,
            group=group.to(device),
            tau=tau,
            self_mask=True,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(proj_g.parameters()) + list(proj_m.parameters()),
                grad_clip
            )
        opt.step()

        loss_ema = loss_ema * 0.9 + float(loss.item()) * 0.1
        pbar.set_postfix({"loss": f"{loss_ema:.4f}"})

    return float(loss_ema)


def _save_ckpt(
    out_path: Path,
    proj_g: nn.Module,
    proj_m: nn.Module,
    cfg,
    fold: int,
    stage: int,
    best_val: float,
    best_epoch_global: int,
) -> None:
    ckpt = {
        "proj_g": proj_g.state_dict(),
        "proj_m": proj_m.state_dict(),
        "config": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg),
        "fold": fold,
        "stage": stage,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch_global),
    }
    torch.save(ckpt, out_path)


def _load_ckpt(
    ckpt_path: Path,
    proj_g: nn.Module,
    proj_m: nn.Module,
    device: torch.device,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    proj_g.load_state_dict(ckpt["proj_g"])
    proj_m.load_state_dict(ckpt["proj_m"])
    return ckpt


def _parse_metric_k(metric: str) -> Optional[int]:
    """
    Supports: hit@10, recall@20, precision@30, mrr@25
    Returns k as int if present, else None.
    """
    m = re.match(r"^(hit|recall|precision|mrr)@(\d+)$", metric.strip())
    if not m:
        return None
    return int(m.group(2))

def run_stage(
    *,
    stage: int,
    proj_g: nn.Module,
    proj_m: nn.Module,
    dl: DataLoader,
    G: torch.Tensor,
    M: torch.Tensor,
    g2true_val: Optional[Dict[int, set]],
    device: torch.device,
    fold: int,
    cfg,
    fold_dir: Path,
    start_epoch_global: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    key_best: Optional[str],     # checkpoint selection metric (None => no best selection)
    key_es: Optional[str],       # early stopping metric (None => no early stop metric)
    eval_topks: Optional[List[int]],
    early_stop: bool,
    patience: int,
    min_delta: float,
    warmup_epochs_global: int,
    ckpt_name: str,              # in full-train, pass "last_*.pth"
) -> Tuple[List[Dict[str, float]], float, int, int, Optional[Path]]:
    """
    Returns:
      history_rows, best_val(key_best), best_epoch_global, end_epoch_global, best_ckpt_path

    Notes:
      - If g2true_val is None (or empty) OR eval_every<=0, evaluation is skipped.
      - In that case, best_ckpt_path remains None, and a LAST checkpoint is saved at stage end.
    """
    opt = torch.optim.AdamW(
        list(proj_g.parameters()) + list(proj_m.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    history_rows: List[Dict[str, float]] = []
    best_val = -1.0
    best_epoch_global = -1
    best_ckpt_path: Optional[Path] = None

    patience_left = int(patience)
    last_best_for_es = -1e18  # comparator for early stop

    # ---- decide whether we can/should evaluate ----
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
            proj_g=proj_g,
            proj_m=proj_m,
            dl=dl,
            G=G,
            M=M,
            device=device,
            tau=float(cfg.tau),
            grad_clip=float(cfg.grad_clip),
            opt=opt,
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

        eval_every = int(getattr(cfg, "eval_every", 0))
        do_eval = (
            (g2true_val is not None)
            and (len(g2true_val) > 0)
            and (eval_topks is not None)
            and (eval_every > 0)
            and (key_best is not None)
            and (key_es is not None)
            )
        evaluated = do_eval and (epoch_global % eval_every == 0)
        if evaluated:
            metrics = eval_retrieval_gly_to_mesh(
                proj_g=proj_g,
                proj_m=proj_m,
                G=G,
                M=M,
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

            # ---- checkpoint selection ----
            if score_best > best_val:
                best_val = score_best
                best_epoch_global = epoch_global

                out_path = fold_dir / ckpt_name
                _save_ckpt(
                    out_path=out_path,
                    proj_g=proj_g,
                    proj_m=proj_m,
                    cfg=cfg,
                    fold=fold,
                    stage=stage,
                    best_val=best_val,
                    best_epoch_global=best_epoch_global,
                )
                best_ckpt_path = out_path
                print(f"  ↳ saved best model to {out_path} (best {key_best}={best_val:.4f} @E{best_epoch_global})")

            # ---- early stopping (on eval steps) ----
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

                    # Save the LAST checkpoint even on early stop (safety net)
                    out_path_last = fold_dir / ckpt_name
                    _save_ckpt(
                        out_path=out_path_last,
                        proj_g=proj_g,
                        proj_m=proj_m,
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

    # ---- always save LAST checkpoint at end of stage ----
    out_path_last = fold_dir / ckpt_name
    _save_ckpt(
        out_path=out_path_last,
        proj_g=proj_g,
        proj_m=proj_m,
        cfg=cfg,
        fold=fold,
        stage=stage,
        best_val=best_val,
        best_epoch_global=best_epoch_global,
    )
    print(f"  ↳ saved LAST model to {out_path_last} (stage{stage} end @E{end_epoch_global})")

    return history_rows, best_val, best_epoch_global, end_epoch_global, best_ckpt_path


# ----------------------------
# 7) Data IO helpers (for reuse in CV / Optuna)
# ----------------------------
def load_embeddings_from_csv(
    gly_emb_csv: str,
    mesh_emb_csv: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      G, M (L2-normalized torch tensors on CPU),
      gly_ids, mesh_ids (as str lists),
      gly_id2row, mesh_id2row
    """
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

    G = l2_normalize(torch.from_numpy(G_np))
    M = l2_normalize(torch.from_numpy(M_np))

    return G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row


def build_pairs_from_csv(
    pairs_csv: str,
    gly_id2row: Dict[str, int],
    mesh_id2row: Dict[str, int],
    gly_id_col: str = "glytoucan_ac",
    mesh_list_col: str = "descriptor_ui_list",
    mesh_sep: str = ";",
) -> Tuple[List[Tuple[int, int]], int, int]:
    """
    pairs_csv row format:
      gly_id_col: glycan id (matches gly embedding id column)
      mesh_list_col: semicolon-separated mesh IDs (matches mesh embedding id column)

    Returns:
      pairs (gi, mi), missing_gly_count, missing_mesh_count
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

        mesh_ids_list = [s.strip() for s in str(mesh_list).split(mesh_sep) if s.strip()]
        for mid in mesh_ids_list:
            mi = mesh_id2row.get(mid, None)
            if mi is None:
                missing_mesh += 1
                continue
            pairs.append((gi, mi))

    return pairs, missing_gly, missing_mesh


def make_kfold_assignment_on_unique_glycans(
    pairs: List[Tuple[int, int]],
    n_folds: int,
    seed: int,
) -> Tuple[Dict[int, int], List[int]]:
    """
    Returns:
      row2fold: glycan_row_index -> fold_id
      gly_in_pairs: sorted list of unique glycan row indices included in pairs
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

def load_preassigned_folds(
    *,
    fold_csv: str,
    gly_id2row: Dict[str, int],
    fold_id_col: str = "graph_id",
    fold_col: str = "cluster_5",
) -> Tuple[Dict[int, int], int]:
    """
    fold_csv: [fold_id_col, fold_col]
      - fold_id_col
      - fold_col   : 0..(n_folds-1) 

    Returns:
      row2fold: gly_row_index -> fold_id
      missing_in_fold_csv: fold_csv
    """
    fold_df = pd.read_csv(fold_csv)

    id2fold = dict(
        zip(
            fold_df[fold_id_col].astype(str).tolist(),
            fold_df[fold_col].astype(int).tolist(),
        )
    )

    row2fold: Dict[int, int] = {}
    missing = 0
    for gid, row_idx in gly_id2row.items():
        f = id2fold.get(str(gid), None)
        if f is None:
            missing += 1
            continue
        row2fold[int(row_idx)] = int(f)

    return row2fold, int(missing)


__all__ = [
    # seed / math
    "set_seed", "l2_normalize", "cosine_sim",
    # model
    "ProjectionHead",
    # dataset / collate
    "GlyMeshPairsDataset", "BatchConfig", "make_collate_fn",
    # loss
    "build_pos_mask", "build_p_distribution", "stablerep_loss",
    # eval
    "eval_retrieval_gly_to_mesh",
    # train helpers
    "_train_one_epoch", "_save_ckpt", "_load_ckpt", "_parse_metric_k", "run_stage",
    # data io
    "load_embeddings_from_csv", "build_pairs_from_csv", "make_kfold_assignment_on_unique_glycans", "load_preassigned_folds",
]
