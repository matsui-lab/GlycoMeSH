#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_cross_modal_stablerep_full_train.py
=========================================

Full-data single-model training for cross-modal multi-positive contrastive learning
(StableRep-style, no negative reweight).

Two-stage training (coarse -> fine), NO early stopping, NO validation evaluation.

Outputs
-------
out_dir/
  full_train/
    history.csv
    last_joint_emb_stablerep_stage1.pth
    last_joint_emb_stablerep_stage2.pth
    last_joint_emb_stablerep_full.pth   (alias of stage2 last)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local reusable utilities (same directory)
from utils import (
    set_seed,
    ProjectionHead,
    GlyMeshPairsDataset,
    BatchConfig,
    make_collate_fn,
    run_stage,
    _load_ckpt,
    load_embeddings_from_csv,
    build_pairs_from_csv,
)


def train(cfg) -> None:
    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
        else ("cpu" if cfg.device == "auto" else cfg.device)
    )
    print("Device:", device)

    # # Enforce global epochs = 45
    # assert cfg.epochs_stage1 + cfg.epochs_stage2 == 45, \
    #     f"Require epochs_stage1 + epochs_stage2 == 45, got {cfg.epochs_stage1} + {cfg.epochs_stage2}"

    # ---- load embeddings (CPU tensors) ----
    G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row = load_embeddings_from_csv(
        cfg.gly_emb_csv, cfg.mesh_emb_csv
    )

    # ---- load labels (pairs_csv) -> pairs (gi, mi) ----
    pairs, missing_gly, missing_mesh = build_pairs_from_csv(
        pairs_csv=cfg.pairs_csv,
        gly_id2row=gly_id2row,
        mesh_id2row=mesh_id2row,
        gly_id_col=cfg.gly_id_col,
        mesh_list_col=cfg.mesh_list_col,
        mesh_sep=cfg.mesh_sep,
    )

    if not pairs:
        raise ValueError(
            "No valid pairs after ID mapping. Check that gly/mesh IDs match embedding CSV first column."
        )
    print(f"[pairs] built={len(pairs)} missing_gly={missing_gly} missing_mesh={missing_mesh}")

    # ---- Full-data training: use ALL pairs ----
    pairs_train: List[Tuple[int, int]] = pairs

    # outputs
    save_dir = Path(cfg.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / "full_train"
    run_dir.mkdir(parents=True, exist_ok=True)

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

    # init model (single)
    proj_g = ProjectionHead(G.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)
    proj_m = ProjectionHead(M.size(1), cfg.out_dim, cfg.hidden, cfg.dropout).to(device)

    history_rows: List[Dict[str, float]] = []

    # NOTE: No validation. We pass g2true_val=None and modify utils.run_stage
    # to skip eval/best/early-stop when g2true_val is None.

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
        g2true_val=None,               # <-- IMPORTANT (no val)
        device=device,
        fold=-1,                       # <-- dummy
        cfg=cfg,
        fold_dir=run_dir,              # <-- output to full_train/
        start_epoch_global=0,
        n_epochs=cfg.epochs_stage1,
        lr=cfg.lr_stage1,
        weight_decay=cfg.weight_decay,
        key_best=None,                 # <-- no best selection
        key_es=None,                   # <-- no early stopping
        eval_topks=None,               # <-- no eval
        early_stop=False,              # <-- enforce no early stopping
        patience=0,
        min_delta=0.0,
        warmup_epochs_global=0,
        ckpt_name="last_joint_emb_stablerep_stage1.pth",   # <-- rename to last
    )
    history_rows.extend(s1_rows)

    # Decide init for stage2
    stage2_start_epoch = s1_end_epoch
    print("[full-train] Stage2 init: continue from Stage1 LAST weights (no validation mode).")

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
        g2true_val=None,               # <-- IMPORTANT (no val)
        device=device,
        fold=-1,                       # <-- dummy
        cfg=cfg,
        fold_dir=run_dir,
        start_epoch_global=stage2_start_epoch,
        n_epochs=cfg.epochs_stage2,
        lr=cfg.lr_stage2,
        weight_decay=cfg.weight_decay,
        key_best=None,                 # <-- no best selection
        key_es=None,                   # <-- no early stopping
        eval_topks=None,               # <-- no eval
        early_stop=False,              # <-- enforce no early stopping
        patience=0,
        min_delta=0.0,
        warmup_epochs_global=0,
        ckpt_name="last_joint_emb_stablerep_stage2.pth",
    )
    history_rows.extend(s2_rows)

    # write history
    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"[full-train] wrote {hist_path}")

    # Optional: alias "full" last checkpoint to stage2 last path (copy)
    # If you prefer not to copy, just keep stage2 as the final.
    # We'll write a small metadata file-like path print instead.
    print("[full-train] final checkpoint:", (run_dir / "last_joint_emb_stablerep_stage2.pth"))


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

    # Two-stage optimization (global epochs must sum to 45)
    ap.add_argument("--lr_stage1", type=float, default=3e-4)
    ap.add_argument("--epochs_stage1", type=int, default=5)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)
    ap.add_argument("--epochs_stage2", type=int, default=40)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Multi-positive batch
    ap.add_argument("--batch_glycans", type=int, default=1024)
    ap.add_argument("--pos_per_glycan", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    # Eval args are kept for compatibility but unused in full-train
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_batch", type=int, default=2048)

    # Early stopping args are kept but unused (forced OFF)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--es_metric", type=str, default="recall@25")
    ap.add_argument("--patience_stage1", type=int, default=0)
    ap.add_argument("--patience_stage2", type=int, default=0)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--warmup_epochs", type=int, default=0)

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", required=True)

    # CV args are kept for CLI compatibility but unused
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--val_fold", type=int, default=-1)

    # best metric args unused
    ap.add_argument("--best_k", type=int, default=25)
    ap.add_argument("--best_metric", type=str, default="auto")

    return ap


if __name__ == "__main__":
    cfg = build_argparser().parse_args()
    train(cfg)
    
# python train_cross_modal_stablerep_full_train.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --out_dir ./data/analysis/contrastive_learning/final_full_model_filtered \
#   --lr_stage1 0.00028541562638670137 \
#   --lr_stage2 0.00009346640154484898 \
#   --tau 0.07441534316265151 \
#   --dropout 0.0025862961419745485 \
#   --weight_decay 0.000132518439144595 \
#   --pos_per_glycan 16 \
#   --batch_glycans 256 \
#   --out_dim 512 \
#   --hidden 1024 \
#   --epochs_stage1 50 \
#   --epochs_stage2 30 \
#   --device auto \
#   --seed 42 \
#   --num_workers 0 \
#   --grad_clip 1.0 \
#   --eval_every 0