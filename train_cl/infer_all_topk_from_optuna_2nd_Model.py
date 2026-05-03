#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_all_topk_from_optuna_2nd_Model.py
-----------------
For each glycan, compute cosine vs all MeSH and write only TopK hits.
Streaming friendly.

Example:
python infer_all_topk_from_optuna_2nd_Model.py \
  --study_dir ./data/analysis/contrastive_learning/optuna_sapbert_A_filtered \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --topk 50 \
  --chunk_g 2048 \
  --out_csv ./data/analysis/contrastive_learning/optuna_sapbert_A_filtered/2nd_model/infer_all_glycan_topk.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import ProjectionHead, load_embeddings_from_csv

SAPBERT_MESH_EMB_CSV = "./data/mesh/embedding/sapbert_name_cls_filtered.csv"


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_proj_from_ckpt(ckpt_path: Path, in_dim_g: int, in_dim_m: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", None)
    if cfg is None:
        cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise ValueError(f"ckpt has no 'config' or 'cfg': {ckpt_path}")

    out_dim = int(cfg["out_dim"])
    hidden = int(cfg["hidden"])
    dropout = float(cfg["dropout"])

    proj_g = ProjectionHead(in_dim_g, out_dim, hidden, dropout).to(device)
    proj_m = ProjectionHead(in_dim_m, out_dim, hidden, dropout).to(device)
    proj_g.load_state_dict(ckpt["proj_g"], strict=True)
    proj_m.load_state_dict(ckpt["proj_m"], strict=True)
    proj_g.eval()
    proj_m.eval()
    return proj_g, proj_m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--mesh_emb_csv", default=SAPBERT_MESH_EMB_CSV)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="auto")

    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk_g", type=int, default=2048)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    study_dir = Path(args.study_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt.strip() else (study_dir / "final_dev_train" / "final_ckpt_stage2.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device(args.device)

    G, M, gly_ids_all, mesh_ids_all, _, _ = load_embeddings_from_csv(args.gly_emb_csv, args.mesh_emb_csv)
    proj_g, proj_m = load_proj_from_ckpt(ckpt_path, G.size(1), M.size(1), device)

    Zm = F.normalize(proj_m(M.to(device)), p=2, dim=1)  # (n_mesh, d_out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topk = max(1, min(int(args.topk), M.size(0)))
    chunk = int(args.chunk_g)

    first = True
    for s in range(0, G.size(0), chunk):
        e = min(G.size(0), s + chunk)
        Zg = F.normalize(proj_g(G[s:e].to(device)), p=2, dim=1)  # (bs, d_out)
        cos = (Zg @ Zm.T).detach().cpu().numpy()  # (bs, n_mesh)

        rows = []
        for i in range(e - s):
            scores = cos[i]
            idx = np.argpartition(-scores, topk - 1)[:topk]
            idx = idx[np.argsort(-scores[idx])]
            gid = gly_ids_all[s + i]
            for r, j in enumerate(idx, start=1):
                rows.append({"gly_id": gid, "rank": r, "mesh_id": mesh_ids_all[j], "cosine": float(scores[j])})

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
        first = False
        print(f"[append] glycans {s}:{e} / {G.size(0)}")

    print("[ok] wrote:", str(out_path))


if __name__ == "__main__":
    main()