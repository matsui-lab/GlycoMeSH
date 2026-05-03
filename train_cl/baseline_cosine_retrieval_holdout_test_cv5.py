#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
baseline_cosine_retrieval_holdout_test_cv5.py
========================================================

Baseline (NO contrastive learning), but split logic is IDENTICAL to CL Strategy A.

- Uses raw glycan embeddings (G) and raw MeSH embeddings (M)
- Similarity = cosine (or dot)
- Metrics identical to CL: hit@k / recall@k / precision@k / mrr@k (glycan->mesh retrieval)

Strategy A (shared with CL):
  1) Fix glycan-level TEST split (10-20%) ONCE (never touched by dev).
  2) On DEV, run 5-fold CV using the SAME fold assignment logic as CL
     (make_kfold_assignment_on_unique_glycans).
     For each fold: evaluate on val glycans.
  3) Final baseline evaluation on fixed TEST.

Outputs (under --out_dir):
  split_test_glycans.json
  dev_cv_fold_metrics.csv
  dev_cv_summary.csv
  test_metrics.json

Run example:
  python baseline_cosine_retrieval_holdout_test_cv5.py \
    --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
    --pairs_csv   ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
    --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
    --out_dir ./data/analysis/baselines \
    --test_frac 0.15 --test_seed 123 \
    --seed 42 \
    --n_folds 5 \
    --topks 1,10,15,20,25,30 \
    --sim cosine \
    --device auto --batch_size 2048

Notes:
- Uses utils.load_embeddings_from_csv + utils.build_pairs_from_csv
  so that ID mapping and candidate MeSH space match CL behavior.
- Candidate MeSH space is ALL rows in mesh_emb_csv (same as CL).
- Split is done at glycan-level on UNIQUE glycans present in `pairs`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch

# Local reusable utilities (same as CL)
from utils import (
    load_embeddings_from_csv,
    build_pairs_from_csv,
    make_kfold_assignment_on_unique_glycans,
    _parse_metric_k,   # only to ensure topks contain requested ks if needed
)


# -------------------------
# Baseline retrieval metrics (same definitions as CL)
# -------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def eval_retrieval_baseline(
    *,
    G: torch.Tensor,                      # (N,D) gly embeddings (CPU tensor ok)
    M: torch.Tensor,                      # (M,D) mesh embeddings (CPU tensor ok)
    g2true: Dict[int, Set[int]],          # gly_row -> set(mesh_row)
    device: torch.device,
    batch_size: int,
    topks: List[int],
    sim: str,
) -> Dict[str, float]:
    """
    Compute hit/recall/precision/mrr @k for glycan->mesh retrieval, identical to CL semantics.
    - For each glycan, rank ALL mesh candidates (rows in M).
    - Use ground truth set (mesh rows) from g2true.
    """
    assert sim in ("cosine", "dot")
    topks = sorted(set(int(k) for k in topks))
    max_k = max(topks)

    M_d = M.to(device)
    if sim == "cosine":
        M_d = l2_normalize(M_d)
    M_T = M_d.t().contiguous()  # (D, M)

    hit = {k: 0 for k in topks}
    rec = {k: 0.0 for k in topks}
    prec = {k: 0.0 for k in topks}
    mrr = {k: 0.0 for k in topks}

    gly_rows = sorted(g2true.keys())
    n_eval = 0

    for s in range(0, len(gly_rows), batch_size):
        batch_rows = gly_rows[s:s + batch_size]
        g_batch = G[batch_rows].to(device)
        if sim == "cosine":
            g_batch = l2_normalize(g_batch)

        scores = g_batch @ M_T  # (B, n_mesh)
        _, idx = torch.topk(scores, k=max_k, dim=1)  # (B, max_k)
        idx = idx.detach().cpu().numpy()

        for bi, g_row in enumerate(batch_rows):
            true_set = set(g2true[g_row])
            if len(true_set) == 0:
                continue
            n_eval += 1
            denom_true = max(1, len(true_set))

            top_all = idx[bi].tolist()
            for k in topks:
                topk = top_all[:k]
                inter = len(set(topk).intersection(true_set))

                # hit@k
                if inter > 0:
                    hit[k] += 1

                # recall@k, precision@k
                rec[k] += inter / denom_true
                prec[k] += inter / float(k)

                # mrr@k (first relevant in top-k)
                rr = 0.0
                for rank, midx in enumerate(topk, start=1):
                    if midx in true_set:
                        rr = 1.0 / rank
                        break
                mrr[k] += rr

    denom = max(1, n_eval)
    out: Dict[str, float] = {"n_eval": float(n_eval)}
    for k in topks:
        out[f"hit@{k}"] = hit[k] / denom
        out[f"recall@{k}"] = rec[k] / denom
        out[f"precision@{k}"] = prec[k] / denom
        out[f"mrr@{k}"] = mrr[k] / denom
    return out


# -------------------------
# Fixed TEST split (SHARED with CL logic)
# -------------------------
def make_fixed_test_split_by_pairs(
    *,
    pairs: List[Tuple[int, int]],
    test_frac: float,
    seed: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Exactly the same style as CL Strategy A:
      - split on UNIQUE glycan rows appearing in pairs (glycan-level)
      - deterministic by (test_frac, seed)
    Return: (test_glycans_set, dev_glycans_set)
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
    n_test = min(len(gly_in_pairs) - 1, n_test)

    test_idx = set(int(i) for i in perm[:n_test].tolist())
    test_g = set(gly_in_pairs[i] for i in test_idx)
    dev_g = set(gly_in_pairs) - test_g
    return test_g, dev_g


def filter_pairs_by_glycans(pairs: List[Tuple[int, int]], keep_glycans: Set[int]) -> List[Tuple[int, int]]:
    return [(g, m) for (g, m) in pairs if g in keep_glycans]


def build_g2true_from_pairs(pairs: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """
    pairs: list of (gly_row, mesh_row)
    return g2true mapping for retrieval evaluation.
    """
    g2true = defaultdict(set)
    for g, m in pairs:
        g2true[g].add(m)
    return {int(g): set(int(x) for x in ms) for g, ms in g2true.items()}


# -------------------------
# CLI
# -------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    # data (match CL)
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--mesh_emb_csv", required=True)

    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")

    ap.add_argument("--out_dir", required=True)

    # split (match CL Strategy A)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--test_seed", type=int, default=123)

    # fold assignment seed (match CL: --seed)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)

    # retrieval
    ap.add_argument("--topks", type=str, default="1,10,15,20,25,30")
    ap.add_argument("--sim", type=str, default="cosine", choices=["cosine", "dot"])

    # compute
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--device", type=str, default="auto")

    return ap


def main():
    args = build_argparser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    topks = sorted(set(topks))
    if len(topks) == 0:
        raise ValueError("--topks is empty")

    # ---- Load embeddings (match CL behavior) ----
    # NOTE: utils.load_embeddings_from_csv expects gly_emb_csv and mesh_emb_csv.
    # Here mesh_emb_csv is explicitly given (not fixed), to support SAPBERT/BioBERT etc.
    G, M, gly_ids, mesh_ids, gly_id2row, mesh_id2row = load_embeddings_from_csv(
        args.gly_emb_csv, args.mesh_emb_csv
    )

    if G.size(1) != M.size(1):
        raise ValueError(f"Dim mismatch: gly {G.size(1)} vs mesh {M.size(1)}")

    print(f"[emb] glycans={G.size(0)} dim={G.size(1)}")
    print(f"[emb] mesh(all)={M.size(0)} dim={M.size(1)}")

    # ---- Load labeled pairs (match CL behavior) ----
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

    # ---- Fixed TEST split (shared logic) ----
    test_glycans, dev_glycans = make_fixed_test_split_by_pairs(
        pairs=pairs, test_frac=float(args.test_frac), seed=int(args.test_seed)
    )
    dev_pairs = filter_pairs_by_glycans(pairs, dev_glycans)
    test_pairs = filter_pairs_by_glycans(pairs, test_glycans)

    if len(dev_pairs) == 0 or len(test_pairs) == 0:
        raise RuntimeError(f"Split produced empty DEV or TEST. dev_pairs={len(dev_pairs)} test_pairs={len(test_pairs)}")

    # save split with explicit glycan IDs
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
        "test_glycan_ids": [str(gly_ids[int(x)]) for x in sorted(test_glycans)],
        "mesh_candidate_size_all": int(M.size(0)),
        "mesh_emb_csv": str(args.mesh_emb_csv),
        "gly_emb_csv": str(args.gly_emb_csv),
        "pairs_csv": str(args.pairs_csv),
    }
    (out_dir / "split_test_glycans.json").write_text(json.dumps(split_out, indent=2), encoding="utf-8")
    print(f"[split] DEV pairs={len(dev_pairs)} TEST pairs={len(test_pairs)} "
          f"DEV glycans={len(dev_glycans)} TEST glycans={len(test_glycans)}")
    print(f"[split] wrote: {(out_dir / 'split_test_glycans.json').as_posix()}")

    # ---- DEV fold assignment (shared logic) ----
    row2fold, gly_in_pairs_dev = make_kfold_assignment_on_unique_glycans(
        pairs=dev_pairs,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
    )
    fold_ids = list(range(int(args.n_folds)))

    # ---- DEV 5-fold evaluation (val glycans per fold) ----
    fold_metrics: List[Dict[str, float]] = []
    for fold in fold_ids:
        # build val pairs by fold
        pairs_val = [(g, m) for (g, m) in dev_pairs if row2fold[g] == fold]
        if len(pairs_val) == 0:
            # Should not happen, but keep safe
            continue

        g2true_val = build_g2true_from_pairs(pairs_val)

        met = eval_retrieval_baseline(
            G=G,
            M=M,
            g2true=g2true_val,
            device=device,
            batch_size=int(args.batch_size),
            topks=topks,
            sim=str(args.sim),
        )

        row = {
            "fold": float(fold),
            "n_eval": float(met.get("n_eval", 0.0)),
            "n_val_pairs": float(len(pairs_val)),
            "n_val_glycans": float(len(g2true_val)),
        }
        for k, v in met.items():
            if k == "n_eval":
                continue
            row[k] = float(v)
        fold_metrics.append(row)

        show_k = max(topks)
        rk = f"recall@{show_k}"
        if rk in met:
            print(f"[DEV fold {fold}] n_eval={int(met['n_eval'])} {rk}={met[rk]:.6f}")

    df_folds = pd.DataFrame(fold_metrics)
    df_folds.to_csv(out_dir / "dev_cv_fold_metrics.csv", index=False)
    print(f"[dev-cv] wrote: {(out_dir / 'dev_cv_fold_metrics.csv').as_posix()}")

    # summary mean/std across folds (like you'd plot error bars)
    metric_cols = [c for c in df_folds.columns if c not in ("fold", "n_eval", "n_val_pairs", "n_val_glycans")]
    summary_rows = []
    for c in metric_cols:
        summary_rows.append({
            "metric": c,
            "mean": float(df_folds[c].mean()),
            "std": float(df_folds[c].std(ddof=1)) if len(df_folds) > 1 else 0.0,
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "dev_cv_summary.csv", index=False)
    print(f"[dev-cv] wrote: {(out_dir / 'dev_cv_summary.csv').as_posix()}")

    # ---- Fixed TEST evaluation (single result) ----
    g2true_test = build_g2true_from_pairs(test_pairs)
    test_metrics = eval_retrieval_baseline(
        G=G,
        M=M,
        g2true=g2true_test,
        device=device,
        batch_size=int(args.batch_size),
        topks=topks,
        sim=str(args.sim),
    )

    test_out = {
        "baseline": {
            "type": "raw_embedding_similarity",
            "sim": str(args.sim),
            "candidate_mesh_space": "ALL",
            "mesh_emb_csv": str(args.mesh_emb_csv),
        },
        "fixed_test": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
            "n_val_pairs_test": int(len(test_pairs)),
            "n_eval_glycans_test": int(test_metrics.get("n_eval", 0)),
        },
        "topks": topks,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_out, indent=2), encoding="utf-8")
    print(f"[test] wrote: {(out_dir / 'test_metrics.json').as_posix()}")

    for k in topks:
        rk, mk = f"recall@{k}", f"mrr@{k}"
        if rk in test_metrics and mk in test_metrics:
            print(f"[TEST] {rk}={test_metrics[rk]:.6f}  {mk}={test_metrics[mk]:.6f}")

    print("[done] Baseline (shared split) complete.")


if __name__ == "__main__":
    main()
