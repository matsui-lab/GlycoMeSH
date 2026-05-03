#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-modal multi-positive contrastive learning (StableRep-style, no negative reweight)
====================================================================================

Two-stage training (coarse -> fine) + optional early stopping.

Stage 1: higher LR to find a good region
Stage 2: lower  LR to refine from the best Stage 1 checkpoint (or last, configurable)

Evaluation Metrics
------------------
For each k in topks, compute:
  - hit@k
  - recall@k
  - precision@k
  - mrr@k

Default topks: [1, 10, 15, 20, 25, 30]

CV
--
Random 5-Fold CV (glycan-level split; shuffle=True, random_state=seed):
  - Split is performed over UNIQUE glycan rows that appear in labeled pairs.
  - train: glycans not in current fold
  - val  : glycans in current fold

Outputs
-------
out_dir/
  cv_summary.csv
  fold_0/
    history.csv
    best_joint_emb_stablerep_stage1.pth
    best_joint_emb_stablerep_stage2.pth
  fold_1/
    ...
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

        g_list, m_list, group_list = [], [], []

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
        scores = (zg_all[g_idx : g_idx + 1] @ zmT).squeeze(0)
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
        "config": vars(cfg),
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
    Supports: hit@10, recall@20, precision@30
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
    g2true_val: Dict[int, set],
    device: torch.device,
    fold: int,
    cfg,
    fold_dir: Path,
    start_epoch_global: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    key_best: str,     # checkpoint selection metric
    key_es: str,       # early stopping metric
    eval_topks: List[int],
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

    for i in range(1, n_epochs + 1):
        epoch_global = start_epoch_global + i

        loss_ema = _train_one_epoch(
            proj_g=proj_g,
            proj_m=proj_m,
            dl=dl,
            G=G,
            M=M,
            device=device,
            tau=cfg.tau,
            grad_clip=cfg.grad_clip,
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

        evaluated = (epoch_global % cfg.eval_every == 0)
        if evaluated:
            metrics = eval_retrieval_gly_to_mesh(
                proj_g=proj_g,
                proj_m=proj_m,
                G=G,
                M=M,
                g2true=g2true_val,
                device=device,
                batch_size=cfg.eval_batch,
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
            if early_stop and epoch_global >= warmup_epochs_global:
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
                    return history_rows, best_val, best_epoch_global, epoch_global, best_ckpt_path

        history_rows.append(row)

    end_epoch_global = start_epoch_global + n_epochs
    return history_rows, best_val, best_epoch_global, end_epoch_global, best_ckpt_path


# ----------------------------
# 7) Training loop with fold CV + two-stage training
# ----------------------------
def train(cfg):
    set_seed(cfg.seed)
    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
        else ("cpu" if cfg.device == "auto" else cfg.device)
    )
    print("Device:", device)

    # ---- load embeddings from CSV (id + dims...) ----
    gly_df = pd.read_csv(cfg.gly_emb_csv)
    mesh_df = pd.read_csv(cfg.mesh_emb_csv)

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

    # ---- load labels (pairs_csv) ----
    pairs_df = pd.read_csv(cfg.pairs_csv)

    pairs: List[Tuple[int, int]] = []
    missing_gly = 0
    missing_mesh = 0

    for _, row in pairs_df.iterrows():
        gly_id = row.get(cfg.gly_id_col, None)
        mesh_list = row.get(cfg.mesh_list_col, None)
        if pd.isna(gly_id) or pd.isna(mesh_list):
            continue

        gi = gly_id2row.get(str(gly_id), None)
        if gi is None:
            missing_gly += 1
            continue

        mesh_ids_list = [s.strip() for s in str(mesh_list).split(cfg.mesh_sep) if s.strip()]
        for mid in mesh_ids_list:
            mi = mesh_id2row.get(mid, None)
            if mi is None:
                missing_mesh += 1
                continue
            pairs.append((gi, mi))

    if not pairs:
        raise ValueError("No valid pairs after ID mapping. Check that gly/mesh IDs match embedding CSV first column.")
    print(f"[pairs] built={len(pairs)} missing_gly={missing_gly} missing_mesh={missing_mesh}")

    # ---- Random KFold assignment on UNIQUE glycans that appear in pairs ----
    gly_in_pairs = sorted({g for (g, _) in pairs})
    if len(gly_in_pairs) < cfg.n_folds:
        raise ValueError(
            f"Not enough unique glycans for n_folds={cfg.n_folds}. "
            f"unique_glycans_in_pairs={len(gly_in_pairs)}"
        )

    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    row2fold: Dict[int, int] = {}
    for fold_id, (_, val_idx) in enumerate(kf.split(gly_in_pairs)):
        for j in val_idx:
            g_row = gly_in_pairs[j]
            row2fold[g_row] = int(fold_id)

    # sanity
    if len(row2fold) != len(gly_in_pairs):
        raise RuntimeError("KFold assignment failed: some glycans did not receive a fold id.")

    # choose folds to run
    fold_ids = list(range(cfg.n_folds))
    if cfg.val_fold >= 0:
        if not (0 <= cfg.val_fold < cfg.n_folds):
            raise ValueError(f"--val_fold must be in [0, {cfg.n_folds-1}] or -1, got {cfg.val_fold}")
        fold_ids = [cfg.val_fold]

    # outputs
    save_dir = Path(cfg.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_fold_summaries: List[Dict[str, float]] = []

    # base eval ks
    eval_topks = [1, 10, 15, 20, 25, 30]

    # checkpoint metric (default: mrr@best_k)
    key_best = cfg.best_metric
    if key_best == "auto":
        key_best = f"mrr@{cfg.best_k}"

    # early stopping metric (default: recall@25)
    key_es = cfg.es_metric

    # ensure eval_topks contains k required by best/es metric
    for metric in [key_best, key_es]:
        k = _parse_metric_k(metric)
        if k is not None and k not in eval_topks:
            eval_topks.append(k)
    eval_topks = sorted(set(eval_topks))

    for fold in fold_ids:
        print("\n====================")
        print(f"Fold {fold}/{cfg.n_folds - 1}")
        print("====================")

        # split by fold (glycan-level)
        pairs_train = [(g, m) for (g, m) in pairs if row2fold[g] != fold]
        pairs_val = [(g, m) for (g, m) in pairs if row2fold[g] == fold]

        if not pairs_train:
            raise ValueError(f"Fold {fold}: no training pairs.")
        if not pairs_val:
            print(f"[warn] Fold {fold}: no validation pairs. Skipping fold.")
            continue

        # build val truth sets
        g2true_val = defaultdict(set)
        for g, m in pairs_val:
            g2true_val[g].add(m)

        # train dataset/loader
        ds_train = GlyMeshPairsDataset(pairs_train)
        batch_cfg = BatchConfig(
            batch_glycans=cfg.batch_glycans,
            pos_per_glycan=cfg.pos_per_glycan,
            allow_pos_replacement=True,
        )
        collate_fn = make_collate_fn(ds_train, batch_cfg)

        dl = DataLoader(
            ds_train,
            batch_size=cfg.batch_glycans,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

        # per-fold dir
        fold_dir = save_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # init model per fold (once, then stage2 continues from stage1 choice)
        proj_g = ProjectionHead(G.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)
        proj_m = ProjectionHead(M.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)

        fold_history: List[Dict[str, float]] = []

        # ------------------
        # Stage 1 (coarse)
        # ------------------
        s1_rows, s1_best, s1_best_epoch, s1_end_epoch, s1_best_path = run_stage(
            stage=1,
            proj_g=proj_g,
            proj_m=proj_m,
            dl=dl,
            G=G,
            M=M,
            g2true_val=g2true_val,
            device=device,
            fold=fold,
            cfg=cfg,
            fold_dir=fold_dir,
            start_epoch_global=0,
            n_epochs=cfg.epochs_stage1,
            lr=cfg.lr_stage1,
            weight_decay=cfg.weight_decay,
            key_best=key_best,
            key_es=key_es,
            eval_topks=eval_topks,
            early_stop=cfg.early_stop,
            patience=cfg.patience_stage1,
            min_delta=cfg.min_delta,
            warmup_epochs_global=cfg.warmup_epochs,
            ckpt_name="best_joint_emb_stablerep_stage1.pth",
        )
        fold_history.extend(s1_rows)

        # Decide init for stage2
        stage2_start_epoch = s1_end_epoch

        if cfg.stage2_init == "best":
            if s1_best_path is None:
                print(f"[fold {fold}] Stage1 produced no best checkpoint; Stage2 will continue from last weights.")
            else:
                _load_ckpt(s1_best_path, proj_g, proj_m, device)
                print(f"[fold {fold}] Stage2 init from Stage1 BEST: {s1_best_path}")
        elif cfg.stage2_init == "last":
            print(f"[fold {fold}] Stage2 init from Stage1 LAST weights.")
        else:
            raise ValueError(f"Unknown --stage2_init={cfg.stage2_init} (use 'best' or 'last')")

        # ------------------
        # Stage 2 (fine)
        # ------------------
        s2_rows, s2_best, s2_best_epoch, s2_end_epoch, s2_best_path = run_stage(
            stage=2,
            proj_g=proj_g,
            proj_m=proj_m,
            dl=dl,
            G=G,
            M=M,
            g2true_val=g2true_val,
            device=device,
            fold=fold,
            cfg=cfg,
            fold_dir=fold_dir,
            start_epoch_global=stage2_start_epoch,
            n_epochs=cfg.epochs_stage2,
            lr=cfg.lr_stage2,
            weight_decay=cfg.weight_decay,
            key_best=key_best,
            key_es=key_es,
            eval_topks=eval_topks,
            early_stop=cfg.early_stop,
            patience=cfg.patience_stage2,
            min_delta=cfg.min_delta,
            warmup_epochs_global=cfg.warmup_epochs,
            ckpt_name="best_joint_emb_stablerep_stage2.pth",
        )
        fold_history.extend(s2_rows)

        # choose fold-best across both stages (by key_best)
        fold_best_val = s2_best if (s2_best >= s1_best) else s1_best
        fold_best_epoch = s2_best_epoch if (s2_best >= s1_best) else s1_best_epoch
        fold_best_stage = 2 if (s2_best >= s1_best) else 1

        # write per-fold history
        hist_df = pd.DataFrame(fold_history)
        hist_path = fold_dir / "history.csv"
        hist_df.to_csv(hist_path, index=False)
        print(f"[fold {fold}] wrote {hist_path} (best {key_best}={fold_best_val:.4f} @E{fold_best_epoch}, stage={fold_best_stage})")

        all_fold_summaries.append({
            "fold": fold,
            "best_stage": fold_best_stage,
            "best_epoch": fold_best_epoch,
            "best_metric": key_best,
            key_best: float(fold_best_val),
            "early_stop_metric": key_es,
            "n_train_pairs": len(pairs_train),
            "n_val_pairs": len(pairs_val),
            "n_val_glycans": len(g2true_val),
            "stage1_best_epoch": s1_best_epoch,
            "stage1_best": float(s1_best),
            "stage2_best_epoch": s2_best_epoch,
            "stage2_best": float(s2_best),
            "stage2_init": cfg.stage2_init,
            "n_unique_glycans_in_pairs": len(gly_in_pairs),
        })

    # write CV summary
    summary_df = pd.DataFrame(all_fold_summaries)
    summary_path = save_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[cv] wrote {summary_path}")
    print(summary_df)


# ----------------------------
# 8) CLI
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gly_emb_csv", required=True, help="CSV: first col glycan_id, rest embedding dims")
    ap.add_argument("--mesh_emb_csv", required=True, help="CSV: first col mesh_id, rest embedding dims")

    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: gly_id, mesh_list (semicolon separated)")
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")

    # Projection
    ap.add_argument("--out_dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Loss
    ap.add_argument("--tau", type=float, default=0.1)

    # Two-stage optimization
    ap.add_argument("--lr_stage1", type=float, default=3e-4)
    ap.add_argument("--epochs_stage1", type=int, default=50)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)
    ap.add_argument("--epochs_stage2", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument(
        "--stage2_init",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Initialize stage2 from stage1 best checkpoint or last weights",
    )

    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Multi-positive batch
    ap.add_argument("--batch_glycans", type=int, default=1024)
    ap.add_argument("--pos_per_glycan", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    # Eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch", type=int, default=2048)

    # Checkpoint selection metric
    ap.add_argument("--best_k", type=int, default=25, help="Used when --best_metric=auto (default mrr@best_k)")
    ap.add_argument(
        "--best_metric",
        type=str,
        default="auto",
        help="Checkpoint selection metric. Use 'auto' (=mrr@best_k) or e.g. 'mrr@30', 'hit@10'.",
    )

    # Early stopping (applies to each stage)
    ap.add_argument("--early_stop", action="store_true", help="Enable early stopping on --es_metric")
    ap.add_argument(
        "--es_metric",
        type=str,
        default="recall@25",
        help="Early stopping metric, e.g. 'hit@10', 'mrr@30', 'recall@20'.",
    )
    ap.add_argument("--patience_stage1", type=int, default=5)
    ap.add_argument("--patience_stage2", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=2, help="Do not early-stop before this global epoch")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", required=True)

    # CV (random KFold)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--val_fold", type=int, default=-1, help="Run only this fold id (0..n_folds-1); -1 runs all folds")

    return ap


if __name__ == "__main__":
    cfg = build_argparser().parse_args()
    train(cfg)
