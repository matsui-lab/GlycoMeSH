#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_all_topk_from_full_glycan_mlp.py
=====================================

For each glycan, run a full-train glycan-only multi-label MLP against the full
candidate MeSH space and write TopK predictions in a streaming-friendly manner.

Important:
- The output dimension corresponds to the MeSH vocabulary order defined by
  --mesh_emb_csv used during training.
- Ranking by logit and ranking by sigmoid(prob) are identical because sigmoid
  is monotonic. Both can be written for convenience.
- Input glycan embedding normalization must match training. The full-train
  script default is L2-normalize unless --no_l2_norm was used.
- To write all predictions, set --topk to the full number of MeSH labels.

Outputs:
  A CSV with columns:
    gly_id, rank, mesh_id[, logit][, prob]

Example (Top-500):
python infer_all_topk_from_full_glycan_mlp.py \
  --study_dir ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --topk 5000 \
  --chunk_g 2048 \
  --device cuda \
  --out_csv ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh/full_train/infer_all_glycan_topk.csv

Example (all MeSH labels):
python infer_all_topk_from_full_glycan_mlp.py \
  --study_dir ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --topk 29382 \
  --chunk_g 2048 \
  --device cuda \
  --out_csv ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh/full_train/infer_all_glycan_allmesh.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from utils_glycan_multilabel import (
    GlycanMLPClassifier,
    load_gly_embeddings_from_csv,
    load_mesh_ids_from_embedding_csv,
)

DEFAULT_MESH_EMB_CSV = "./data/mesh/embedding/sapbert_name_cls_filtered.csv"


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model_from_ckpt(
    ckpt_path: Path,
    in_dim: int,
    n_mesh: int,
    device: torch.device,
) -> Tuple[GlycanMLPClassifier, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = ckpt.get("cfg", None)
    if cfg is None:
        cfg = ckpt.get("config", None)
    if cfg is None:
        raise ValueError(f"Checkpoint has no 'cfg' or 'config': {ckpt_path}")

    required = ["hidden1", "hidden2", "dropout"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Checkpoint cfg missing keys: {missing}")

    model = GlycanMLPClassifier(
        in_dim=int(in_dim),
        out_dim=int(n_mesh),
        hidden1=int(cfg["hidden1"]),
        hidden2=int(cfg["hidden2"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    if "model" not in ckpt:
        raise KeyError(f"Checkpoint has no 'model' state_dict: {ckpt_path}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def infer_topk_streaming(
    *,
    model: GlycanMLPClassifier,
    G: torch.Tensor,
    gly_ids: List[str],
    mesh_ids: List[str],
    out_csv: Path,
    topk: int,
    chunk_g: int,
    device: torch.device,
    write_logit: bool,
    write_prob: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_gly = int(G.size(0))
    n_mesh = int(len(mesh_ids))
    topk = max(1, min(int(topk), n_mesh))
    chunk_g = max(1, int(chunk_g))

    first = True
    for s in range(0, n_gly, chunk_g):
        e = min(n_gly, s + chunk_g)

        x = G[s:e].to(device, non_blocking=True)
        logits = model(x)
        topv, topi = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)

        topi_cpu = topi.detach().cpu()
        topv_cpu = topv.detach().cpu() if write_logit else None
        topp_cpu = torch.sigmoid(topv).detach().cpu() if write_prob else None

        rows = []
        batch_size = e - s
        for i in range(batch_size):
            gid = gly_ids[s + i]
            idx_row = topi_cpu[i].tolist()
            logit_row = topv_cpu[i].tolist() if write_logit else None
            prob_row = topp_cpu[i].tolist() if write_prob else None

            for rank, j in enumerate(idx_row, start=1):
                rec = {
                    "gly_id": gid,
                    "rank": rank,
                    "mesh_id": mesh_ids[j],
                }
                if write_logit:
                    rec["logit"] = float(logit_row[rank - 1])
                if write_prob:
                    rec["prob"] = float(prob_row[rank - 1])
                rows.append(rec)

        df = pd.DataFrame(rows)
        df.to_csv(
            out_csv,
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False
        print(f"[append] glycans {s}:{e} / {n_gly}")

    print(f"[ok] wrote: {out_csv}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--study_dir",
        required=True,
        help="Directory containing full_train/. Used to resolve the default checkpoint path and store metadata references.",
    )
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--mesh_emb_csv", default=DEFAULT_MESH_EMB_CSV)

    ap.add_argument(
        "--ckpt",
        default="",
        help="Optional checkpoint path. Default: <study_dir>/full_train/final_ckpt_stage2.pth",
    )
    ap.add_argument("--device", default="auto")

    ap.add_argument(
        "--no_l2_norm",
        action="store_true",
        help="Disable L2 normalization of input glycan embeddings. Must match training-time preprocessing.",
    )

    ap.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of predictions to write per glycan. Set to the full MeSH count to output all predictions.",
    )
    ap.add_argument("--chunk_g", type=int, default=2048)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--save_meta_json",
        default="",
        help="Optional path to save inference metadata JSON.",
    )

    ap.add_argument("--no_logit", action="store_true", help="Do not write the logit column.")
    ap.add_argument("--no_prob", action="store_true", help="Do not write the prob=sigmoid(logit) column.")

    return ap


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()

    study_dir = Path(args.study_dir)
    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt.strip()
        else (study_dir / "full_train" / "final_ckpt_stage2.pth")
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device(args.device)

    G, gly_ids, _ = load_gly_embeddings_from_csv(
        args.gly_emb_csv,
        normalize=(not args.no_l2_norm),
    )
    mesh_ids, _ = load_mesh_ids_from_embedding_csv(args.mesh_emb_csv)

    model, cfg = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        in_dim=int(G.size(1)),
        n_mesh=int(len(mesh_ids)),
        device=device,
    )

    write_logit = not args.no_logit
    write_prob = not args.no_prob
    if (not write_logit) and (not write_prob):
        raise ValueError("At least one of logit/prob must be written.")

    print(
        "[loaded] "
        f"n_gly={G.size(0)} "
        f"emb_dim={G.size(1)} "
        f"n_mesh={len(mesh_ids)} "
        f"device={device}"
    )
    print(
        "[model] "
        f"hidden1={cfg.get('hidden1')} "
        f"hidden2={cfg.get('hidden2')} "
        f"dropout={cfg.get('dropout')}"
    )

    out_csv = Path(args.out_csv)
    infer_topk_streaming(
        model=model,
        G=G,
        gly_ids=gly_ids,
        mesh_ids=mesh_ids,
        out_csv=out_csv,
        topk=int(args.topk),
        chunk_g=int(args.chunk_g),
        device=device,
        write_logit=write_logit,
        write_prob=write_prob,
    )

    if args.save_meta_json.strip():
        meta = {
            "script": "infer_all_topk_from_full_glycan_mlp.py",
            "study_dir": str(study_dir),
            "ckpt": str(ckpt_path),
            "gly_emb_csv": str(args.gly_emb_csv),
            "mesh_emb_csv": str(args.mesh_emb_csv),
            "normalize_gly_input_l2": bool(not args.no_l2_norm),
            "topk": int(args.topk),
            "chunk_g": int(args.chunk_g),
            "device": str(device),
            "n_gly": int(G.size(0)),
            "n_mesh": int(len(mesh_ids)),
            "write_logit": bool(write_logit),
            "write_prob": bool(write_prob),
            "model_cfg_from_ckpt": cfg,
            "output_csv": str(out_csv),
        }
        meta_path = Path(args.save_meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()

# python infer_all_topk_from_full_glycan_mlp.py \
#   --study_dir ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
#   --topk 5000 \
#   --chunk_g 2048 \
#   --device cuda \
#   --out_csv ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh/full_train/infer_all_glycan_topk.csv