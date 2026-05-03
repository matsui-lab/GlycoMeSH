#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_glycan_mlp_full_train.py
==============================

Full-data single-model training for glycan-only multi-label MeSH prediction.

"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from utils_glycan_multilabel import (
    set_seed,
    GlycanMLPClassifier,
    GlycanMultiLabelDataset,
    load_gly_embeddings_from_csv,
    load_mesh_ids_from_embedding_csv,
    build_pairs_from_csv_with_given_mesh_vocab,
    build_multilabel_targets,
    make_pos_weight,
)


# ----------------------------
# Helpers
# ----------------------------
def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _save_ckpt(path: Path, model: torch.nn.Module, cfg_dict: Dict, extra: Dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "cfg": cfg_dict,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


# ----------------------------
# Full training
# ----------------------------
def train(cfg) -> None:
    set_seed(int(cfg.seed))
    device = _get_device(cfg.device)
    print("Device:", device)

    out_dir = Path(cfg.out_dir)
    run_dir = out_dir / "full_train"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- load glycan embeddings ----
    G, gly_ids, gly_id2row = load_gly_embeddings_from_csv(
        cfg.gly_emb_csv,
        normalize=(not cfg.no_l2_norm),
    )

    # ---- full candidate MeSH vocabulary ----
    mesh_ids, mesh_id2row = load_mesh_ids_from_embedding_csv(cfg.mesh_emb_csv)

    # ---- supervision pairs mapped onto the full MeSH vocab ----
    pairs, missing_gly, missing_mesh = build_pairs_from_csv_with_given_mesh_vocab(
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

    # ---- dense multi-label targets over all glycans ----
    Y, _ = build_multilabel_targets(
        pairs=pairs,
        n_gly=G.size(0),
        n_mesh=len(mesh_ids),
    )

    train_gly = sorted({g for (g, _) in pairs})
    ds_train = GlycanMultiLabelDataset(train_gly)
    dl = torch.utils.data.DataLoader(
        ds_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = GlycanMLPClassifier(
        in_dim=G.size(1),
        out_dim=Y.size(1),
        hidden1=int(cfg.hidden1),
        hidden2=int(cfg.hidden2),
        dropout=float(cfg.dropout),
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

    history_rows: List[Dict] = []
    cfg_dict = vars(cfg).copy()

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

            for g_idx in dl:
                x = G[g_idx].to(device, non_blocking=True)
                y = Y[g_idx].to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(cfg.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()

                loss_item = float(loss.item())
                loss_sum += loss_item
                n_batches += 1
                n_seen += int(x.size(0))
                loss_ema = loss_item if loss_ema is None else (loss_ema * 0.9 + loss_item * 0.1)

            row = {
                "stage": int(stage),
                "epoch_in_stage": int(ep),
                "global_epoch": int(sum(r["epoch_in_stage"] for r in history_rows if r["stage"] < stage) + ep),
                "lr": float(lr),
                "train_loss_mean": float(loss_sum / max(n_batches, 1)),
                "train_loss_ema": float(loss_ema if loss_ema is not None else float("nan")),
                "n_batches": int(n_batches),
                "n_seen_glycans": int(n_seen),
            }
            history_rows.append(row)

            print(
                f"[full_train] stage={stage} epoch={ep}/{n_epochs} "
                f"loss_mean={row['train_loss_mean']:.6f} "
                f"loss_ema={row['train_loss_ema']:.6f}"
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

    # stage 2 (continue from stage 1 last)
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
        "model": "glycan_only_multilabel_mlp_full_train",
        "gly_emb_csv": str(cfg.gly_emb_csv),
        "mesh_emb_csv": str(cfg.mesh_emb_csv),
        "pairs_csv": str(cfg.pairs_csv),
        "n_pairs_total": int(len(pairs)),
        "n_unique_glycans_train": int(len(train_gly)),
        "n_mesh_labels": int(len(mesh_ids)),
        "missing_gly": int(missing_gly),
        "missing_mesh": int(missing_mesh),
        "train_params": {
            "hidden1": int(cfg.hidden1),
            "hidden2": int(cfg.hidden2),
            "dropout": float(cfg.dropout),
            "lr_stage1": float(cfg.lr_stage1),
            "lr_stage2": float(cfg.lr_stage2),
            "weight_decay": float(cfg.weight_decay),
            "epochs_stage1": int(cfg.epochs_stage1),
            "epochs_stage2": int(cfg.epochs_stage2),
            "batch_size": int(cfg.batch_size),
            "pos_weight_mode": str(cfg.pos_weight_mode),
            "pos_weight_clip_max": float(cfg.pos_weight_clip_max),
            "grad_clip": float(cfg.grad_clip),
            "seed": int(cfg.seed),
            "device": str(device),
        },
        "artifacts": {
            "history_csv": str((run_dir / "history.csv").as_posix()),
            "stage1_ckpt": str((run_dir / "final_ckpt_stage1.pth").as_posix()),
            "stage2_ckpt": str((run_dir / "final_ckpt_stage2.pth").as_posix()),
            "full_ckpt": str((run_dir / "final_ckpt_full.pth").as_posix()),
        },
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[full_train] wrote:", (run_dir / "history.csv").as_posix())
    print("[full_train] wrote:", (run_dir / "train_summary.json").as_posix())
    print("[full_train] final checkpoint:", (run_dir / "final_ckpt_full.pth").as_posix())


# ----------------------------
# CLI
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--mesh_emb_csv", required=True, help="CSV whose first column is MeSH ID; defines the full candidate MeSH space")
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")
    ap.add_argument("--no_l2_norm", action="store_true", help="Disable L2 normalization of input glycan embeddings")

    # model
    ap.add_argument("--hidden1", type=int, default=1024)
    ap.add_argument("--hidden2", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    # training
    ap.add_argument("--lr_stage1", type=float, default=1e-3)
    ap.add_argument("--lr_stage2", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs_stage1", type=int, default=50)
    ap.add_argument("--epochs_stage2", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
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

# Example:
# python train_glycan_mlp_full_train.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --out_dir ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh \
#   --hidden1 1536 \
#   --hidden2 0 \
#   --dropout 0.3009141678482733 \
#   --lr_stage1 0.004802242732075022 \
#   --lr_stage2 0.00048237737607982373 \
#   --weight_decay 0.0006080635354589146 \
#   --batch_size 64 \
#   --pos_weight_mode balanced \
#   --pos_weight_clip_max 20.0 \
#   --epochs_stage1 50 \
#   --epochs_stage2 30 \
#   --device cuda \
#   --seed 42 \
#   --num_workers 0 \
#   --grad_clip 1.0