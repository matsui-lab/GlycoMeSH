#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_all_topk_from_full_glycan_mesh_dual_mlp.py
================================================

For each glycan, run a full-train glycan+MeSH dual-MLP model against the full
candidate MeSH space and write TopK predictions in a streaming-friendly manner.

Model
-----
Dual encoder:
  glycan embedding --MLP--> joint space
  MeSH embedding   --MLP--> joint space
  score(glycan, mesh) = scale * z_g @ z_m^T
  where scale = exp(logit_scale)

Important
---------
- The output dimension corresponds to the MeSH vocabulary order defined by
  --mesh_emb_csv used during training.
- Ranking by score and ranking by sigmoid(prob) are identical because sigmoid
  is monotonic. Both can be written for convenience.
- Input glycan/MeSH embedding normalization must match training.
  Full-train defaults:
    - glycan embeddings: L2-normalized unless --no_l2_norm_gly was used
    - MeSH embeddings  : L2-normalized unless --no_l2_norm_mesh was used
- Joint-space normalization (normalize_joint) is restored from the checkpoint cfg.
- To write all predictions, set --topk to the full number of MeSH labels.

Outputs
-------
A CSV with columns:
  gly_id, rank, mesh_id[, score][, prob]

Example (Top-5000)
------------------
python infer_all_topk_from_full_glycan_mesh_dual_mlp.py \
  --study_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/final_full_model_run01 \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --topk 5000 \
  --chunk_g 1024 \
  --chunk_m 4096 \
  --device cuda \
  --out_csv ./data/analysis/multilabel_glycan_mesh_dual_mlp/final_full_model_run01/full_train/infer_all_glycan_topk.csv

Example (all MeSH labels)
-------------------------
python infer_all_topk_from_full_glycan_mesh_dual_mlp.py \
  --study_dir ./data/analysis/multilabel_glycan_mesh_dual_mlp/final_full_model_run01 \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --topk 5000 \
  --chunk_g 512 \
  --chunk_m 4096 \
  --device cuda \
  --out_csv ./data/analysis/multilabel_glycan_mesh_dual_mlp/final_full_model_run01/full_train/infer_all_glycan_allmesh.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_MESH_EMB_CSV = "./data/mesh/embedding/sapbert_name_cls_filtered.csv"


# ----------------------------
# 1) Utils
# ----------------------------
def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _coerce_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ----------------------------
# 2) Model
# ----------------------------
def _make_mlp(in_dim: int, hidden1: int, hidden2: int, out_dim: int, dropout: float) -> nn.Sequential:
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
    return nn.Sequential(*layers)


class GlyMeshDualMLP(nn.Module):
    """
    Dual encoder:
      glycan embedding -> MLP -> z_g
      mesh embedding   -> MLP -> z_m
      score = scale * z_g @ z_m^T
    """

    def __init__(
        self,
        gly_in_dim: int,
        mesh_in_dim: int,
        proj_dim: int = 512,
        gly_hidden1: int = 1024,
        gly_hidden2: int = 0,
        mesh_hidden1: int = 1024,
        mesh_hidden2: int = 0,
        dropout: float = 0.1,
        normalize_joint: bool = True,
        init_logit_scale: float = 1.0,
    ):
        super().__init__()
        self.gly_net = _make_mlp(gly_in_dim, gly_hidden1, gly_hidden2, proj_dim, dropout)
        self.mesh_net = _make_mlp(mesh_in_dim, mesh_hidden1, mesh_hidden2, proj_dim, dropout)
        self.normalize_joint = bool(normalize_joint)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(math.log(init_logit_scale)), dtype=torch.float32)
        )

    def encode_gly(self, x_g: torch.Tensor) -> torch.Tensor:
        z = self.gly_net(x_g)
        if self.normalize_joint:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def encode_mesh(self, x_m: torch.Tensor) -> torch.Tensor:
        z = self.mesh_net(x_m)
        if self.normalize_joint:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def score_from_encoded(self, z_g: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1e-3, max=100.0)
        return scale * (z_g @ z_m.T)


# ----------------------------
# 3) IO helpers
# ----------------------------
def load_embeddings_from_csv(
    gly_emb_csv: str,
    mesh_emb_csv: str,
    normalize_gly: bool = True,
    normalize_mesh: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], Dict[str, int], Dict[str, int]]:
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

    G = torch.from_numpy(G_np)
    M = torch.from_numpy(M_np)

    if normalize_gly:
        G = l2_normalize(G)
    if normalize_mesh:
        M = l2_normalize(M)

    return G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row


def load_model_from_ckpt(
    ckpt_path: Path,
    gly_in_dim: int,
    mesh_in_dim: int,
    device: torch.device,
) -> Tuple[GlyMeshDualMLP, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = ckpt.get("cfg", None)
    if cfg is None:
        cfg = ckpt.get("config", None)
    if cfg is None:
        raise ValueError(f"Checkpoint has no 'cfg' or 'config': {ckpt_path}")

    required = [
        "proj_dim",
        "gly_hidden1",
        "gly_hidden2",
        "mesh_hidden1",
        "mesh_hidden2",
        "dropout",
        "normalize_joint",
        "init_logit_scale",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Checkpoint cfg missing keys: {missing}")

    model = GlyMeshDualMLP(
        gly_in_dim=int(gly_in_dim),
        mesh_in_dim=int(mesh_in_dim),
        proj_dim=int(cfg["proj_dim"]),
        gly_hidden1=int(cfg["gly_hidden1"]),
        gly_hidden2=int(cfg["gly_hidden2"]),
        mesh_hidden1=int(cfg["mesh_hidden1"]),
        mesh_hidden2=int(cfg["mesh_hidden2"]),
        dropout=float(cfg["dropout"]),
        normalize_joint=bool(cfg["normalize_joint"]),
        init_logit_scale=float(cfg["init_logit_scale"]),
    ).to(device)

    if "model" not in ckpt:
        raise KeyError(f"Checkpoint has no 'model' state_dict: {ckpt_path}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


# ----------------------------
# 4) Inference
# ----------------------------
@torch.no_grad()
def precompute_mesh_joint(
    *,
    model: GlyMeshDualMLP,
    M: torch.Tensor,
    chunk_m: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Precompute MeSH projected embeddings in the joint space.
    Returns CPU tensor of shape [n_mesh, proj_dim].
    """
    z_parts = []
    n_mesh = int(M.size(0))
    chunk_m = max(1, int(chunk_m))

    for s in range(0, n_mesh, chunk_m):
        e = min(n_mesh, s + chunk_m)
        z_m = model.encode_mesh(M[s:e].to(device, non_blocking=True)).detach().cpu()
        z_parts.append(z_m)
        print(f"[mesh encode] {s}:{e} / {n_mesh}")

    return torch.cat(z_parts, dim=0)


@torch.no_grad()
def infer_topk_streaming(
    *,
    model: GlyMeshDualMLP,
    G: torch.Tensor,
    gly_ids: List[str],
    mesh_ids: List[str],
    z_m_all_cpu: torch.Tensor,
    out_csv: Path,
    topk: int,
    chunk_g: int,
    device: torch.device,
    write_score: bool,
    write_prob: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_gly = int(G.size(0))
    n_mesh = int(len(mesh_ids))
    topk = max(1, min(int(topk), n_mesh))
    chunk_g = max(1, int(chunk_g))

    z_m_all_cpu = z_m_all_cpu.contiguous()
    z_mT_cpu = z_m_all_cpu.T.contiguous()
    scale = model.logit_scale.exp().clamp(min=1e-3, max=100.0).detach().cpu()

    first = True
    for s in range(0, n_gly, chunk_g):
        e = min(n_gly, s + chunk_g)

        x_g = G[s:e].to(device, non_blocking=True)
        z_g_cpu = model.encode_gly(x_g).detach().cpu()

        scores = scale * (z_g_cpu @ z_mT_cpu)
        topv, topi = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)

        topi_cpu = topi
        topv_cpu = topv if write_score else None
        topp_cpu = torch.sigmoid(topv) if write_prob else None

        rows = []
        batch_size = e - s
        for i in range(batch_size):
            gid = gly_ids[s + i]
            idx_row = topi_cpu[i].tolist()
            score_row = topv_cpu[i].tolist() if write_score else None
            prob_row = topp_cpu[i].tolist() if write_prob else None

            for rank, j in enumerate(idx_row, start=1):
                rec = {
                    "gly_id": gid,
                    "rank": rank,
                    "mesh_id": mesh_ids[j],
                }
                if write_score:
                    rec["score"] = float(score_row[rank - 1])
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


# ----------------------------
# 5) CLI
# ----------------------------
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
        "--no_l2_norm_gly",
        action="store_true",
        help="Disable L2 normalization of input glycan embeddings. Must match training-time preprocessing.",
    )
    ap.add_argument(
        "--no_l2_norm_mesh",
        action="store_true",
        help="Disable L2 normalization of input MeSH embeddings. Must match training-time preprocessing.",
    )

    ap.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of predictions to write per glycan. Set to the full MeSH count to output all predictions.",
    )
    ap.add_argument("--chunk_g", type=int, default=1024)
    ap.add_argument("--chunk_m", type=int, default=4096)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--save_meta_json",
        default="",
        help="Optional path to save inference metadata JSON.",
    )

    ap.add_argument("--no_score", action="store_true", help="Do not write the score column.")
    ap.add_argument("--no_prob", action="store_true", help="Do not write the prob=sigmoid(score) column.")

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

    G, M, gly_ids, mesh_ids, _, _ = load_embeddings_from_csv(
        args.gly_emb_csv,
        args.mesh_emb_csv,
        normalize_gly=(not args.no_l2_norm_gly),
        normalize_mesh=(not args.no_l2_norm_mesh),
    )

    model, cfg = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        gly_in_dim=int(G.size(1)),
        mesh_in_dim=int(M.size(1)),
        device=device,
    )

    write_score = not args.no_score
    write_prob = not args.no_prob
    if (not write_score) and (not write_prob):
        raise ValueError("At least one of score/prob must be written.")

    print(
        "[loaded] "
        f"n_gly={G.size(0)} "
        f"gly_dim={G.size(1)} "
        f"n_mesh={len(mesh_ids)} "
        f"mesh_dim={M.size(1)} "
        f"device={device}"
    )
    print(
        "[model] "
        f"proj_dim={cfg.get('proj_dim')} "
        f"gly_hidden1={cfg.get('gly_hidden1')} "
        f"gly_hidden2={cfg.get('gly_hidden2')} "
        f"mesh_hidden1={cfg.get('mesh_hidden1')} "
        f"mesh_hidden2={cfg.get('mesh_hidden2')} "
        f"dropout={cfg.get('dropout')} "
        f"normalize_joint={cfg.get('normalize_joint')} "
        f"init_logit_scale={cfg.get('init_logit_scale')}"
    )

    z_m_all_cpu = precompute_mesh_joint(
        model=model,
        M=M,
        chunk_m=int(args.chunk_m),
        device=device,
    )

    out_csv = Path(args.out_csv)
    infer_topk_streaming(
        model=model,
        G=G,
        gly_ids=gly_ids,
        mesh_ids=mesh_ids,
        z_m_all_cpu=z_m_all_cpu,
        out_csv=out_csv,
        topk=int(args.topk),
        chunk_g=int(args.chunk_g),
        device=device,
        write_score=write_score,
        write_prob=write_prob,
    )

    if args.save_meta_json.strip():
        meta = {
            "script": "infer_all_topk_from_full_glycan_mesh_dual_mlp.py",
            "study_dir": str(study_dir),
            "ckpt": str(ckpt_path),
            "gly_emb_csv": str(args.gly_emb_csv),
            "mesh_emb_csv": str(args.mesh_emb_csv),
            "normalize_gly_input_l2": bool(not args.no_l2_norm_gly),
            "normalize_mesh_input_l2": bool(not args.no_l2_norm_mesh),
            "topk": int(args.topk),
            "chunk_g": int(args.chunk_g),
            "chunk_m": int(args.chunk_m),
            "device": str(device),
            "n_gly": int(G.size(0)),
            "n_mesh": int(len(mesh_ids)),
            "write_score": bool(write_score),
            "write_prob": bool(write_prob),
            "model_cfg_from_ckpt": _coerce_jsonable(cfg),
            "output_csv": str(out_csv),
        }
        meta_path = Path(args.save_meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()