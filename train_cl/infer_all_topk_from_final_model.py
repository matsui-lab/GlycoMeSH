#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_all_topk_from_final_model.py
-----------------
For each glycan, compute cosine vs all MeSH and write only TopK hits.
Streaming friendly.

Example:
python infer_all_topk_from_final_model.py \
  --study_dir ./data/analysis/contrastive_learning/final_full_model_filtered \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large_strict.csv \
  --topk 5000 \
  --chunk_g 2048 \
  --out_csv ./data/analysis/contrastive_learning/final_full_model_filtered/infer_all_glycan_topk.csv \
  --out_gly_z_csv ./data/analysis/contrastive_learning/final_full_model_filtered/gly_emb.csv \
  --out_mesh_z_csv ./data/analysis/contrastive_learning/final_full_model_filtered/mesh_emb.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json

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

def write_emb_csv(path: Path, ids, Z: torch.Tensor, prefix: str, mode: str = "cols", first: bool = True):
    """
    ids: list[str]
    Z: (n, d) torch.Tensor on CPU ok
    mode:
      - "cols"
      - "json"
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    Znp = Z.detach().cpu().numpy().astype(np.float32)
    if mode == "cols":
        d = Znp.shape[1]
        data = {"id": ids}
        for k in range(d):
            data[f"{prefix}_{k}"] = Znp[:, k]
        df = pd.DataFrame(data)
    elif mode == "json":
        df = pd.DataFrame({
            "id": ids,
            "emb": [json.dumps(vec.tolist()) for vec in Znp],
        })
    else:
        raise ValueError(f"unknown mode: {mode}")

    df.to_csv(path, index=False, mode="w" if first else "a", header=first)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--mesh_emb_csv", default=SAPBERT_MESH_EMB_CSV)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="auto")

    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--chunk_g", type=int, default=2048)
    ap.add_argument("--out_csv", required=True)
    
    ap.add_argument("--out_gly_z_csv", required=True)
    ap.add_argument("--out_mesh_z_csv", required=True)
    ap.add_argument("--write_z_as", choices=["cols", "json"], default="cols")
    
    args = ap.parse_args()

    study_dir = Path(args.study_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt.strip() else (study_dir / "full_train" / "last_joint_emb_stablerep_stage2.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device(args.device)

    G, M, gly_ids_all, mesh_ids_all, _, _ = load_embeddings_from_csv(args.gly_emb_csv, args.mesh_emb_csv)
    proj_g, proj_m = load_proj_from_ckpt(ckpt_path, G.size(1), M.size(1), device)

    # MeSH 
    Zm = F.normalize(proj_m(M.to(device)), p=2, dim=1)  # (n_mesh, d_out)
    if args.out_mesh_z_csv.strip():
        write_emb_csv(Path(args.out_mesh_z_csv), mesh_ids_all, Zm, prefix="z", mode=args.write_z_as, first=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topk = max(1, min(int(args.topk), M.size(0)))
    chunk = int(args.chunk_g)

    # glycan 
    write_gly_z = bool(args.out_gly_z_csv.strip())
    gly_z_path = Path(args.out_gly_z_csv) if write_gly_z else None
    gly_first = True

    first = True

    for s in range(0, G.size(0), chunk):
        e = min(G.size(0), s + chunk)
        Zg = F.normalize(proj_g(G[s:e].to(device)), p=2, dim=1)  # (bs, d_out)

        # glycan
        if write_gly_z:
            write_emb_csv(gly_z_path, gly_ids_all[s:e], Zg, prefix="z", mode=args.write_z_as, first=gly_first)
            gly_first = False

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
    if args.out_gly_z_csv.strip():
        print("[ok] wrote gly Z:", str(args.out_gly_z_csv))
    if args.out_mesh_z_csv.strip():
        print("[ok] wrote mesh Z:", str(args.out_mesh_z_csv))


if __name__ == "__main__":
    main()
    
# python infer_all_topk_from_final_model.py   --study_dir ./data/analysis/contrastive_learning/final_full_model_filtered \
    # --gly_emb_csv ./data/glycangt/glycangt_emb_human_rat_mouse.csv   --topk 5000   --chunk_g 2048 \
    # --out_csv ./data/analysis/contrastive_learning/final_full_model_filtered/infer_all_glycan_topk.csv \
    # --out_mesh_z_csv ./data/analysis/contrastive_learning/final_full_model_filtered/mesh_emb.csv \
    # --out_gly_z_csv ./data/analysis/contrastive_learning/final_full_model_filtered/gly_emb.csv