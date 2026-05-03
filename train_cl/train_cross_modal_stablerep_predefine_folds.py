#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_cross_modal_stablerep_predefine_folds.py
==============================================

Cross-modal multi-positive contrastive learning (StableRep-style, no negative reweight)

Two-stage training (coarse -> fine) + optional early stopping.

Adds:
  - optional export of validation top-k predictions + cosine similarity scores
    from the best checkpoint of each stage.

Outputs
-------
out_dir/
  cv_summary.csv
  fold_0/
    history.csv
    best_joint_emb_stablerep_stage1.pth
    best_joint_emb_stablerep_stage2.pth
    val_topk_stage1_best.csv           (optional)
    val_topk_stage2_best.csv           (optional)
  fold_1/
    ...
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils_kfold import (
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
    load_preassigned_folds,
    predict_topk_gly_to_mesh,
)


def _export_val_topk_predictions(
    *,
    proj_g: ProjectionHead,
    proj_m: ProjectionHead,
    G: torch.Tensor,
    M: torch.Tensor,
    gly_ids: List[str],
    mesh_ids: List[str],
    g2true_val: Dict[int, set],
    device: torch.device,
    batch_size: int,
    topk: int,
    out_csv: Path,
) -> None:
    val_gly_indices = sorted(g2true_val.keys())
    pred_df = predict_topk_gly_to_mesh(
        proj_g=proj_g,
        proj_m=proj_m,
        G=G,
        M=M,
        gly_indices=val_gly_indices,
        gly_ids=gly_ids,
        mesh_ids=mesh_ids,
        g2true=g2true_val,
        device=device,
        batch_size=batch_size,
        topk=topk,
    )
    pred_df.to_csv(out_csv, index=False)
    print(f"[pred] wrote {out_csv} (rows={len(pred_df)})")


def train(cfg) -> None:
    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
        else ("cpu" if cfg.device == "auto" else cfg.device)
    )
    print("Device:", device)

    G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row = load_embeddings_from_csv(
        cfg.gly_emb_csv, cfg.mesh_emb_csv
    )

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

    row2fold, missing_fold = load_preassigned_folds(
        fold_csv=cfg.fold_csv,
        gly_id2row=gly_id2row,
        fold_id_col=cfg.fold_id_col,
        fold_col=cfg.fold_col,
    )
    print(f"[fold] mapped={len(row2fold)} missing_in_fold_csv={missing_fold}")

    pairs = [(g, m) for (g, m) in pairs if g in row2fold]
    if not pairs:
        raise ValueError("No pairs remain after filtering by fold assignment. Check fold_csv coverage.")

    fold_ids = list(range(cfg.n_folds))
    if cfg.val_fold >= 0:
        if not (0 <= cfg.val_fold < cfg.n_folds):
            raise ValueError(f"--val_fold must be in [0, {cfg.n_folds - 1}] or -1, got {cfg.val_fold}")
        fold_ids = [cfg.val_fold]

    save_dir = Path(cfg.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_fold_summaries: List[Dict[str, float]] = []

    eval_topks = [1, 10, 15, 20, 25, 30]

    key_best = cfg.best_metric
    if key_best == "auto":
        key_best = f"mrr@{cfg.best_k}"

    key_es = cfg.es_metric

    for metric in [key_best, key_es]:
        k = _parse_metric_k(metric)
        if k is not None and k not in eval_topks:
            eval_topks.append(k)
    eval_topks = sorted(set(eval_topks))

    for fold in fold_ids:
        print("\n====================")
        print(f"Fold {fold}/{cfg.n_folds - 1}")
        print("====================")

        pairs_train: List[Tuple[int, int]] = [(g, m) for (g, m) in pairs if row2fold[g] != fold]
        pairs_val: List[Tuple[int, int]] = [(g, m) for (g, m) in pairs if row2fold[g] == fold]

        if not pairs_train:
            raise ValueError(f"Fold {fold}: no training pairs.")
        if not pairs_val:
            print(f"[warn] Fold {fold}: no validation pairs. Skipping fold.")
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

        dl = DataLoader(
            ds_train,
            batch_size=cfg.batch_glycans,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

        fold_dir = save_dir / f"fold_{fold}"
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

        if cfg.save_topk_predictions:
            stage1_pred_path = fold_dir / "val_topk_stage1_best.csv"
            if s1_best_path is not None:
                _load_ckpt(s1_best_path, proj_g, proj_m, device)
                _export_val_topk_predictions(
                    proj_g=proj_g,
                    proj_m=proj_m,
                    G=G,
                    M=M,
                    gly_ids=gly_ids,
                    mesh_ids=mesh_ids,
                    g2true_val=g2true_val,
                    device=device,
                    batch_size=cfg.eval_batch,
                    topk=cfg.pred_topk,
                    out_csv=stage1_pred_path,
                )
            else:
                print(f"[pred] fold {fold}: stage1 best checkpoint not found; skip stage1 prediction export.")

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

        if cfg.save_topk_predictions:
            stage2_pred_path = fold_dir / "val_topk_stage2_best.csv"
            if s2_best_path is not None:
                _load_ckpt(s2_best_path, proj_g, proj_m, device)
                _export_val_topk_predictions(
                    proj_g=proj_g,
                    proj_m=proj_m,
                    G=G,
                    M=M,
                    gly_ids=gly_ids,
                    mesh_ids=mesh_ids,
                    g2true_val=g2true_val,
                    device=device,
                    batch_size=cfg.eval_batch,
                    topk=cfg.pred_topk,
                    out_csv=stage2_pred_path,
                )
            else:
                print(f"[pred] fold {fold}: stage2 best checkpoint not found; skip stage2 prediction export.")

        fold_best_val = s2_best if (s2_best >= s1_best) else s1_best
        fold_best_epoch = s2_best_epoch if (s2_best >= s1_best) else s1_best_epoch
        fold_best_stage = 2 if (s2_best >= s1_best) else 1

        hist_df = pd.DataFrame(fold_history)
        hist_path = fold_dir / "history.csv"
        hist_df.to_csv(hist_path, index=False)
        print(
            f"[fold {fold}] wrote {hist_path} "
            f"(best {key_best}={fold_best_val:.4f} @E{fold_best_epoch}, stage={fold_best_stage})"
        )

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
            "n_unique_glycans_in_pairs": float(len({g for (g, _) in pairs})),
        })

    summary_df = pd.DataFrame(all_fold_summaries)
    summary_path = save_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[cv] wrote {summary_path}")
    print(summary_df)


def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gly_emb_csv", required=True, help="CSV: first col glycan_id, rest embedding dims")
    ap.add_argument("--mesh_emb_csv", required=True, help="CSV: first col mesh_id, rest embedding dims")

    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: gly_id, mesh_list (semicolon separated)")
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")

    ap.add_argument("--out_dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--tau", type=float, default=0.1)

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

    ap.add_argument("--batch_glycans", type=int, default=1024)
    ap.add_argument("--pos_per_glycan", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch", type=int, default=2048)

    ap.add_argument("--best_k", type=int, default=25, help="Used when --best_metric=auto (default mrr@best_k)")
    ap.add_argument(
        "--best_metric",
        type=str,
        default="auto",
        help="Checkpoint selection metric. Use 'auto' (=mrr@best_k) or e.g. 'mrr@30', 'hit@10', 'recall@25'.",
    )

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

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--val_fold", type=int, default=-1, help="Run only this fold id (0..n_folds-1); -1 runs all folds")

    ap.add_argument("--fold_csv", required=True, help="CSV: fold assignment table (id -> fold)")
    ap.add_argument("--fold_id_col", default="graph_id", help="ID column in fold_csv")
    ap.add_argument("--fold_col", default="cluster_5", help="Fold id column in fold_csv (0..n_folds-1)")

    ap.add_argument("--save_topk_predictions", action="store_true", help="Save validation top-k predictions from best checkpoint of each stage")
    ap.add_argument("--pred_topk", type=int, default=30, help="Number of top predictions to export per validation glycan")

    return ap


if __name__ == "__main__":
    cfg = build_argparser().parse_args()
    train(cfg)
    
# python train_cross_modal_stablerep_predefine_folds.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --out_dir ./data/analysis/contrastive_learning/predifined_folds_filtered \
#   --fold_csv ./data/glycan/graph_id_cluster5_5cv.csv \
#   --fold_id_col graph_id \
#   --fold_col cluster_5 \
#   --n_folds 5 \
#   --device auto \
#   --seed 42 \
#   --lr_stage1 0.00028541562638670137 \
#   --lr_stage2 0.00009346640154484898 \
#   --tau 0.07441534316265151 \
#   --dropout 0.0025862961419745485 \
#   --weight_decay 0.000132518439144595 \
#   --pos_per_glycan 16 \
#   --batch_glycans 256 \
#   --out_dim 512 \
#   --hidden 1024 \
#   --best_metric recall@25 \
#   --es_metric recall@25 \
#   --early_stop \
#   --patience_stage1 5 \
#   --patience_stage2 10 \
#   --min_delta 0.001 \
#   --warmup_epochs 2 \
#   --save_topk_predictions \
#   --pred_topk 30