#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
optuna_sapbert_stablerep_holdout_test_cv5.py
===========================================

Strategy A (most clean; higher compute):
  1) Fix a glycan-level TEST split (10-20%) ONCE, never touched by Optuna/dev.
  2) On DEV (remaining 80-90%), run Optuna where objective = 5-fold CV mean(metric).
  3) Pick best HP.
  4) Retrain on ALL DEV with best HP (two-stage, NO val / NO early-stop by val).
  5) Final evaluation on fixed TEST (report metrics; paper main result).

Target mesh embedding: SAPBERT
  mesh_emb_csv = ./data/mesh/embedding/sapbert_name_cls.csv

Outputs (under --study_dir):
  study.db (SQLite)                 : Optuna persistent storage
  study_best.json                   : best trial summary
  split_test_glycans.json           : fixed test glycan row indices + metadata
  trials/trial_<number>/
    params.json
    cv_summary.csv
    fold_0/ ... fold_4/ (history.csv + checkpoints from CV)
  final_dev_train/
    final_ckpt_stage1.pth
    final_ckpt_stage2.pth
    test_metrics.json

Example:
  python optuna_sapbert_stablerep_holdout.py \
    --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
    --pairs_csv   ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
    --study_dir   ./data/analysis/contrastive_learning/optuna_sapbert_A_filtered \
    --n_trials 100 \
    --metric recall@25 \
    --test_frac 0.15 \
    --early_stop \
    --prune

Resume (same command, same study_dir/study_name):
  python optuna_sapbert_stablerep_holdout_test_cv5.py ...

Notes:
- TEST split is determined ONLY by (--test_frac, --test_seed) and glycan-level IDs
  (row indices in embedding table). Keep them fixed across runs for reproducibility.
- During Optuna objective, we run DEV 5-fold CV each trial. Heavy but clean.
- Final DEV retrain uses NO validation and does not peek at TEST until the end.

"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import optuna
from optuna.trial import Trial

# Local reusable utilities (same directory)
from utils import (
    set_seed,
    ProjectionHead,
    GlyMeshPairsDataset,
    BatchConfig,
    make_collate_fn,
    run_stage,
    _load_ckpt,
    _parse_metric_k,
    load_embeddings_from_csv,
    build_pairs_from_csv,
    make_kfold_assignment_on_unique_glycans,
    eval_retrieval_gly_to_mesh,
    stablerep_loss,
)


SAPBERT_MESH_EMB_CSV = "./data/mesh/embedding/sapbert_name_cls_filtered.csv"


# ----------------------------
# Config holder
# ----------------------------
@dataclass
class TrainCfg:
    # loss / projection
    tau: float
    out_dim: int
    hidden: int
    dropout: float

    # optimization (two-stage)
    lr_stage1: float
    lr_stage2: float
    weight_decay: float
    epochs_stage1: int
    epochs_stage2: int
    stage2_init: str  # "best" or "last"
    grad_clip: float

    # batching
    batch_glycans: int
    pos_per_glycan: int
    num_workers: int

    # eval / selection (for CV only)
    eval_every: int
    eval_batch: int
    best_k: int
    best_metric: str
    es_metric: str
    early_stop: bool
    patience_stage1: int
    patience_stage2: int
    min_delta: float
    warmup_epochs: int

    # misc
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
# Fixed TEST split (glycan-level)
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
    n_test = min(len(gly_in_pairs) - 1, n_test)  # keep at least 1 in dev

    test_idx = set(int(i) for i in perm[:n_test].tolist())
    test_g = set(gly_in_pairs[i] for i in test_idx)
    dev_g = set(gly_in_pairs) - test_g
    return test_g, dev_g


def filter_pairs_by_glycans(pairs: List[Tuple[int, int]], keep_glycans: Set[int]) -> List[Tuple[int, int]]:
    return [(g, m) for (g, m) in pairs if g in keep_glycans]


# ----------------------------
# CV on DEV (objective)
# ----------------------------
def _run_dev_cv5_once(
    *,
    cfg: TrainCfg,
    G: torch.Tensor,
    M: torch.Tensor,
    dev_pairs: List[Tuple[int, int]],
    out_dir: Path,
    metric: str,
    trial: Optional[Trial] = None,
    row2fold: Dict[int, int],         
) -> Tuple[float, pd.DataFrame]:
    """
    Run 5-fold CV on DEV pairs for a single hyperparameter configuration.
    Objective value = mean(metric across folds) where metric == key_best by default.
    """
    set_seed(cfg.seed)
    device = _get_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fold_ids = list(range(cfg.n_folds))
    gly_in_pairs = sorted({g for (g, _) in dev_pairs})

    # base eval ks
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
        pairs_train = [(g, m) for (g, m) in dev_pairs if row2fold[g] != fold]
        pairs_val = [(g, m) for (g, m) in dev_pairs if row2fold[g] == fold]
        if not pairs_train or not pairs_val:
            # Shouldn't happen, but guard.
            continue

        g2true_val = defaultdict(set)
        for g, m in pairs_val:
            g2true_val[g].add(m)

        ds_train = GlyMeshPairsDataset(pairs_train)
        batch_cfg = BatchConfig(
            batch_glycans=cfg.batch_glycans,
            pos_per_glycan=cfg.pos_per_glycan,
            allow_pos_replacement=True,
        )
        collate_fn = make_collate_fn(ds_train, batch_cfg)

        dl = torch.utils.data.DataLoader(
            ds_train,
            batch_size=cfg.batch_glycans,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        proj_g = ProjectionHead(G.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)
        proj_m = ProjectionHead(M.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)

        fold_history: List[Dict[str, float]] = []

        # Stage 1
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

        # Stage 2 init
        stage2_start_epoch = s1_end_epoch
        if cfg.stage2_init == "best" and s1_best_path is not None:
            _load_ckpt(s1_best_path, proj_g, proj_m, device)

        # Stage 2
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

        # fold best by key_best
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
            "n_train_pairs": float(len(pairs_train)),
            "n_val_pairs": float(len(pairs_val)),
            "n_val_glycans": float(len(g2true_val)),
            "stage1_best_epoch": float(s1_best_epoch),
            "stage1_best": float(s1_best),
            "stage2_best_epoch": float(s2_best_epoch),
            "stage2_best": float(s2_best),
            "stage2_init": str(cfg.stage2_init),
            "n_unique_glycans_in_pairs": float(len(gly_in_pairs)),
        })

        # pruning hook: after each fold report mean so far
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
# Final retrain on DEV (no val), then evaluate on TEST
# ----------------------------
@torch.no_grad()
def _eval_on_test(
    *,
    proj_g: ProjectionHead,
    proj_m: ProjectionHead,
    G: torch.Tensor,
    M: torch.Tensor,
    test_pairs: List[Tuple[int, int]],
    device: torch.device,
    topks: List[int],
) -> Dict[str, float]:
    g2true_test = defaultdict(set)
    for g, m in test_pairs:
        g2true_test[g].add(m)
    metrics = eval_retrieval_gly_to_mesh(
        proj_g=proj_g,
        proj_m=proj_m,
        G=G,
        M=M,
        g2true=g2true_test,
        device=device,
        batch_size=2048,
        topks=topks,
    )
    return {k: float(v) for k, v in metrics.items()}


def _train_two_stage_no_val(
    *,
    cfg: TrainCfg,
    G: torch.Tensor,
    M: torch.Tensor,
    dev_pairs: List[Tuple[int, int]],
    out_dir: Path,
    device: torch.device,
) -> Tuple[ProjectionHead, ProjectionHead]:
    """
    Two-stage training on ALL DEV pairs without any validation metric selection.
    Saves final stage checkpoints (last weights) for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = GlyMeshPairsDataset(dev_pairs)
    batch_cfg = BatchConfig(
        batch_glycans=cfg.batch_glycans,
        pos_per_glycan=cfg.pos_per_glycan,
        allow_pos_replacement=True,
    )
    collate_fn = make_collate_fn(ds_train, batch_cfg)

    dl = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg.batch_glycans,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    proj_g = ProjectionHead(G.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)
    proj_m = ProjectionHead(M.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)

    def train_stage(lr: float, n_epochs: int, stage_name: str) -> None:
        opt = torch.optim.AdamW(
            list(proj_g.parameters()) + list(proj_m.parameters()),
            lr=float(lr),
            weight_decay=float(cfg.weight_decay),
        )
        for ep in range(1, n_epochs + 1):
            # simple epoch loop (no eval)
            proj_g.train()
            proj_m.train()
            loss_ema = 0.0
            for g_idx, m_idx, group in dl:
                g_emb = G[g_idx].to(device)
                m_emb = M[m_idx].to(device)
                zg = proj_g(g_emb)
                zm = proj_m(m_emb)
                loss = stablerep_loss(
                    z_g=zg,
                    z_m=zm,
                    group=group.to(device),
                    tau=float(cfg.tau),
                    self_mask=True,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(cfg.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(proj_g.parameters()) + list(proj_m.parameters()),
                        float(cfg.grad_clip),
                    )
                opt.step()
                loss_ema = loss_ema * 0.9 + float(loss.item()) * 0.1
            print(f"[final_dev_train] {stage_name} epoch={ep}/{n_epochs} train_loss_ema={loss_ema:.6f}")

    # Stage 1
    train_stage(cfg.lr_stage1, cfg.epochs_stage1, "stage1")
    torch.save(
        {"proj_g": proj_g.state_dict(), "proj_m": proj_m.state_dict(), "cfg": cfg.__dict__},
        out_dir / "final_ckpt_stage1.pth",
    )

    # Stage 2 init
    # (stage2_init only meaningful when stage1 has "best" ckpt; here we always continue from last)
    train_stage(cfg.lr_stage2, cfg.epochs_stage2, "stage2")
    torch.save(
        {"proj_g": proj_g.state_dict(), "proj_m": proj_m.state_dict(), "cfg": cfg.__dict__},
        out_dir / "final_ckpt_stage2.pth",
    )

    return proj_g, proj_m


# ----------------------------
# Optuna search space (SAPBERT narrowed)
# ----------------------------
def suggest_sapbert_params(trial: Trial) -> Dict:
    """
    Local search around a given good configuration (nearby exploration).
    Center:
      lr_stage1 ~ 3.98e-4
      lr_stage2 ~ 6.11e-5
      tau       ~ 0.0537
      dropout   ~ 0.019
      wd        ~ 2.55e-4
      pos_per_glycan=16, batch_glycans=256, out_dim=512, hidden=1024
    """
    lr_stage1 = trial.suggest_float("lr_stage1", 2.5e-4, 6.5e-4, log=True)
    lr_stage2 = trial.suggest_float("lr_stage2", 2.0e-5, 1.2e-4, log=True)
    tau       = trial.suggest_float("tau",       0.030,  0.080,  log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.06)
    weight_decay = trial.suggest_float("weight_decay", 5.0e-6, 2.0e-3, log=True)

    pos_per_glycan = trial.suggest_categorical("pos_per_glycan", [16])
    batch_glycans = trial.suggest_categorical("batch_glycans", [256])
    out_dim = trial.suggest_categorical("out_dim", [512])
    hidden = trial.suggest_categorical("hidden", [1024])

    return dict(
        lr_stage1=float(lr_stage1),
        lr_stage2=float(lr_stage2),
        tau=float(tau),
        dropout=float(dropout),
        weight_decay=float(weight_decay),
        pos_per_glycan=int(pos_per_glycan),
        batch_glycans=int(batch_glycans),
        out_dim=int(out_dim),
        hidden=int(hidden),
    )


def objective_factory(
    args,
    G: torch.Tensor,
    M: torch.Tensor,
    dev_pairs: List[Tuple[int, int]],
    gly_ids: List[str],
    row2fold_dev: Dict[int, int]
):
    metric = args.metric

    def objective(trial: Trial) -> float:
        trial_seed = (args.seed * 1000003 + trial.number) & 0xFFFFFFFF
        hp = suggest_sapbert_params(trial)

        best_metric = args.best_metric
        if best_metric == "auto":
            best_metric = metric  # keep signal consistent

        cfg = TrainCfg(
            tau=float(hp["tau"]),
            out_dim=int(hp["out_dim"]),
            hidden=int(hp["hidden"]),
            dropout=float(hp["dropout"]),

            lr_stage1=float(hp["lr_stage1"]),
            lr_stage2=float(hp["lr_stage2"]),
            weight_decay=float(hp["weight_decay"]),
            epochs_stage1=int(args.epochs_stage1),
            epochs_stage2=int(args.epochs_stage2),
            stage2_init=str(args.stage2_init),
            grad_clip=float(args.grad_clip),

            batch_glycans=int(hp["batch_glycans"]),
            pos_per_glycan=int(hp["pos_per_glycan"]),
            num_workers=int(args.num_workers),

            eval_every=int(args.eval_every),
            eval_batch=int(args.eval_batch),
            best_k=int(args.best_k),
            best_metric=str(best_metric),
            es_metric=str(args.es_metric),
            early_stop=bool(args.early_stop),
            patience_stage1=int(args.patience_stage1),
            patience_stage2=int(args.patience_stage2),
            min_delta=float(args.min_delta),
            warmup_epochs=int(args.warmup_epochs),

            seed=int(trial_seed),
            device=str(args.device),
            n_folds=int(args.n_folds),
        )

        trial_dir = Path(args.study_dir) / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        params_out = {
            "mesh_embedding": "sapbert",
            "mesh_emb_csv": SAPBERT_MESH_EMB_CSV,
            "trial_number": int(trial.number),
            "trial_seed": int(trial_seed),
            "objective_metric": str(metric),
            "params": hp,
            "fixed": {
                "epochs_stage1": int(args.epochs_stage1),
                "epochs_stage2": int(args.epochs_stage2),
                "n_folds": int(args.n_folds),
                "test_frac": float(args.test_frac),
                "test_seed": int(args.test_seed),
            },
        }
        (trial_dir / "params.json").write_text(json.dumps(params_out, indent=2), encoding="utf-8")

        val, _ = _run_dev_cv5_once(
            cfg=cfg,
            G=G,
            M=M,
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
# CLI
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")

    # fixed test split
    ap.add_argument("--test_frac", type=float, default=0.15, help="Fixed test fraction (glycan-level), e.g. 0.1-0.2")
    ap.add_argument("--test_seed", type=int, default=123, help="Seed to deterministically create fixed test split")

    # study
    ap.add_argument("--study_dir", required=True, help="Directory for sqlite + trial outputs")
    ap.add_argument("--study_name", default="sapbert_stablerep_holdout_test_cv5")
    ap.add_argument("--storage", default="", help="Override optuna storage URL. Default uses sqlite in study_dir.")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--timeout_min", type=int, default=0, help="0 disables timeout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_seed", type=int, default=42, help="Seed for DEV 5-fold split (fixed across trials/runs). Should match stage1 CV seed, typically 42.",)

    # objective metric
    ap.add_argument("--metric", default="recall@25")
    ap.add_argument("--best_metric", default="auto", help="CV checkpoint selection metric. Default=objective metric.")
    ap.add_argument("--best_k", type=int, default=25)

    # schedule
    ap.add_argument("--epochs_stage1", type=int, default=50)
    ap.add_argument("--epochs_stage2", type=int, default=100)
    ap.add_argument("--stage2_init", type=str, default="best", choices=["best", "last"])

    # early stopping (CV only)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--es_metric", default="recall@25")
    ap.add_argument("--patience_stage1", type=int, default=5)
    ap.add_argument("--patience_stage2", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    # batch / eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch", type=int, default=2048)
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

    # optuna storage
    storage = args.storage.strip() if args.storage.strip() else f"sqlite:///{(study_dir / 'study.db').as_posix()}"

    # load embeddings (mesh fixed to SAPBERT)
    G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row = load_embeddings_from_csv(
        args.gly_emb_csv, SAPBERT_MESH_EMB_CSV
    )

    # load pairs
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
    print(f"[pairs] built={len(pairs)} missing_gly={missing_gly} missing_mesh={missing_mesh}")

    # fixed TEST split (glycan-level)
    test_glycans, dev_glycans = make_fixed_test_split(
        pairs=pairs, test_frac=float(args.test_frac), seed=int(args.test_seed)
    )
    dev_pairs = filter_pairs_by_glycans(pairs, dev_glycans)
    test_pairs = filter_pairs_by_glycans(pairs, test_glycans)

    if not dev_pairs or not test_pairs:
        raise RuntimeError(f"Split produced empty DEV or TEST. dev_pairs={len(dev_pairs)} test_pairs={len(test_pairs)}")
    
    # ----------------------------
    # Fixed DEV 5-fold assignment (saved once)
    # ----------------------------
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
        "test_glycan_rows": sorted(int(x) for x in test_glycans),
    }
    (study_dir / "split_test_glycans.json").write_text(json.dumps(split_out, indent=2), encoding="utf-8")
    print(f"[split] DEV pairs={len(dev_pairs)} TEST pairs={len(test_pairs)} "
          f"DEV glycans={len(dev_glycans)} TEST glycans={len(test_glycans)}")

    # pruner
    if args.prune:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(args.startup_trials)) if args.pruner == "median" else optuna.pruners.NopPruner()
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

    objective = objective_factory(args, G, M, dev_pairs, gly_ids, row2fold_dev)

    timeout = None if int(args.timeout_min) <= 0 else int(args.timeout_min) * 60

    t0 = time.time()
    study.optimize(objective, n_trials=int(args.n_trials), timeout=timeout, gc_after_trial=True)
    t1 = time.time()

    print(f"[optuna done] elapsed_sec={t1 - t0:.1f}")
    print("Best value (DEV CV mean):", study.best_value)
    print("Best params:", study.best_params)

    # write best (dev-cv)
    best_out = {
        "study_name": args.study_name,
        "storage": storage,
        "objective_metric": args.metric,
        "best_value_dev_cv_mean": float(study.best_value),
        "best_params": dict(study.best_params),
        "mesh_embedding": "sapbert",
        "mesh_emb_csv": SAPBERT_MESH_EMB_CSV,
        "test_split": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
        },
    }
    (study_dir / "study_best.json").write_text(json.dumps(best_out, indent=2), encoding="utf-8")

    # ----------------------------
    # Final: retrain on ALL DEV with best HP, then evaluate on fixed TEST
    # ----------------------------
    print("\n[final] Retrain on ALL DEV with best HP (no val), then evaluate on fixed TEST.")

    # Build final cfg using best params; keep training schedule from CLI; disable early_stop for final
    bp = dict(study.best_params)
    final_cfg = TrainCfg(
        tau=float(bp["tau"]),
        out_dim=int(bp["out_dim"]),
        hidden=int(bp["hidden"]),
        dropout=float(bp["dropout"]),

        lr_stage1=float(bp["lr_stage1"]),
        lr_stage2=float(bp["lr_stage2"]),
        weight_decay=float(bp["weight_decay"]),
        epochs_stage1=int(args.epochs_stage1),
        epochs_stage2=int(args.epochs_stage2),
        stage2_init=str(args.stage2_init),
        grad_clip=float(args.grad_clip),

        batch_glycans=int(bp["batch_glycans"]),
        pos_per_glycan=int(bp["pos_per_glycan"]),
        num_workers=int(args.num_workers),

        # unused in final (no val), but keep consistent
        eval_every=int(args.eval_every),
        eval_batch=int(args.eval_batch),
        best_k=int(args.best_k),
        best_metric=str(args.best_metric if args.best_metric != "auto" else args.metric),
        es_metric=str(args.es_metric),
        early_stop=False,
        patience_stage1=int(args.patience_stage1),
        patience_stage2=int(args.patience_stage2),
        min_delta=float(args.min_delta),
        warmup_epochs=int(args.warmup_epochs),

        seed=int(args.seed),  # deterministic final
        device=str(args.device),
        n_folds=int(args.n_folds),
    )

    device = _get_device(final_cfg.device)
    final_dir = study_dir / "final_dev_train"
    proj_g, proj_m = _train_two_stage_no_val(
        cfg=final_cfg,
        G=G,
        M=M,
        dev_pairs=dev_pairs,
        out_dir=final_dir,
        device=device,
    )

    # final test metrics
    # ensure topks contains objective metric k
    eval_topks = [1, 10, 15, 20, 25, 30]
    for mname in [args.metric]:
        k = _parse_metric_k(mname)
        if k is not None and k not in eval_topks:
            eval_topks.append(k)
    eval_topks = sorted(set(eval_topks))

    test_metrics = _eval_on_test(
        proj_g=proj_g,
        proj_m=proj_m,
        G=G,
        M=M,
        test_pairs=test_pairs,
        device=device,
        topks=eval_topks,
    )

    test_out = {
        "mesh_embedding": "sapbert",
        "mesh_emb_csv": SAPBERT_MESH_EMB_CSV,
        "objective_metric": str(args.metric),
        "best_params": dict(study.best_params),
        "final_dev_train": {
            "epochs_stage1": int(args.epochs_stage1),
            "epochs_stage2": int(args.epochs_stage2),
            "batch_glycans": int(final_cfg.batch_glycans),
            "pos_per_glycan": int(final_cfg.pos_per_glycan),
            "seed": int(final_cfg.seed),
        },
        "fixed_test": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
            "n_pairs_test": int(len(test_pairs)),
            "n_unique_glycans_test": int(len(test_glycans)),
        },
        "test_metrics": test_metrics,
    }
    (final_dir / "test_metrics.json").write_text(json.dumps(test_out, indent=2), encoding="utf-8")

    print("[final] wrote:", (final_dir / "test_metrics.json").as_posix())
    print("[final] TEST metric summary:")
    # show a compact line
    keys_show = [f"hit@{k}" for k in eval_topks] + [f"recall@{k}" for k in eval_topks] + [f"mrr@{k}" for k in eval_topks]
    keys_show = [k for k in keys_show if k in test_metrics]
    for k in keys_show:
        print(f"  {k}: {test_metrics[k]:.6f}")

    print("\n[done] Strategy A complete.")
    print("  - Optuna best on DEV CV mean saved to study_best.json")
    print("  - Final DEV retrain + TEST eval saved to final_dev_train/test_metrics.json")


if __name__ == "__main__":
    main()