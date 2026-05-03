#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
global_cooccurrence_frequency_holdout_test_cv5.py
=================================================

Baseline:
  Global co-occurrence / frequency ranking

Idea
----
For every query glycan, rank MeSH labels only by their frequency in the TRAIN set.

That is, this baseline ignores glycan features entirely and uses only the
global MeSH frequency prior estimated from the training data.

Scoring
-------
  score(m) = count_train(m)

where count_train(m) is the number of train glycan-MeSH pairs containing m.

Strategy
--------
1) Fix a glycan-level TEST split ONCE
2) On DEV, run fixed 5-fold CV
3) No hyperparameter tuning is needed
4) Evaluate on fixed TEST using all DEV pairs as reference frequency counts

Outputs
-------
study_dir/
  split_test_glycans.json
  fold_assignment_dev.csv
  dev_cv_summary.csv
  final_test_metrics.json

Example
-------
python global_cooccurrence_frequency_holdout_test_cv5.py \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
  --study_dir ./data/analysis/global_cooccurrence_frequency/ \
  --metric recall@25 \
  --test_frac 0.15 \
  --test_seed 123 \
  --cv_seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ----------------------------
# 1) Utilities
# ----------------------------
def _parse_metric_k(metric: str) -> Optional[int]:
    """
    Supports: hit@10, recall@20, precision@30, mrr@25
    """
    m = re.match(r"^(hit|recall|precision|mrr)@(\d+)$", metric.strip())
    if not m:
        return None
    return int(m.group(2))


# ----------------------------
# 2) Data IO
# ----------------------------
def load_gly_ids_from_embedding_csv(
    gly_emb_csv: str,
) -> Tuple[List[str], Dict[str, int]]:
    gly_df = pd.read_csv(gly_emb_csv)
    gly_id_col = gly_df.columns[0]

    gly_ids = gly_df[gly_id_col].astype(str).tolist()
    gly_id2row = {gid: i for i, gid in enumerate(gly_ids)}
    return gly_ids, gly_id2row


def load_mesh_ids_from_pairs_csv(
    pairs_csv: str,
    mesh_list_col: str = "descriptor_ui_list",
    mesh_sep: str = ";",
) -> Tuple[List[str], Dict[str, int]]:
    pairs_df = pd.read_csv(pairs_csv)
    mesh_set = set()

    for _, row in pairs_df.iterrows():
        mesh_list = row.get(mesh_list_col, None)
        if pd.isna(mesh_list):
            continue
        mids = [s.strip() for s in str(mesh_list).split(mesh_sep) if s.strip()]
        mesh_set.update(mids)

    mesh_ids = sorted(mesh_set)
    mesh_id2row = {mid: i for i, mid in enumerate(mesh_ids)}
    return mesh_ids, mesh_id2row


def build_pairs_from_csv(
    pairs_csv: str,
    gly_id2row: Dict[str, int],
    mesh_id2row: Dict[str, int],
    gly_id_col: str = "glytoucan_ac",
    mesh_list_col: str = "descriptor_ui_list",
    mesh_sep: str = ";",
) -> Tuple[List[Tuple[int, int]], int, int]:
    pairs_df = pd.read_csv(pairs_csv)

    pairs: List[Tuple[int, int]] = []
    missing_gly = 0
    missing_mesh = 0

    for _, row in pairs_df.iterrows():
        gly_id = row.get(gly_id_col, None)
        mesh_list = row.get(mesh_list_col, None)
        if pd.isna(gly_id) or pd.isna(mesh_list):
            continue

        gi = gly_id2row.get(str(gly_id), None)
        if gi is None:
            missing_gly += 1
            continue

        mids = [s.strip() for s in str(mesh_list).split(mesh_sep) if s.strip()]
        for mid in mids:
            mi = mesh_id2row.get(mid, None)
            if mi is None:
                missing_mesh += 1
                continue
            pairs.append((gi, mi))

    pairs = sorted(set(pairs))
    return pairs, missing_gly, missing_mesh


# ----------------------------
# 3) Split helpers
# ----------------------------
def make_fixed_test_split(
    *,
    pairs: List[Tuple[int, int]],
    test_frac: float,
    seed: int,
) -> Tuple[Set[int], Set[int]]:
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


def make_kfold_assignment_on_unique_glycans(
    pairs: List[Tuple[int, int]],
    n_folds: int,
    seed: int,
) -> Tuple[Dict[int, int], List[int]]:
    gly_in_pairs = sorted({g for (g, _) in pairs})
    if len(gly_in_pairs) < n_folds:
        raise ValueError(
            f"Not enough unique glycans for n_folds={n_folds}. "
            f"unique_glycans_in_pairs={len(gly_in_pairs)}"
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    row2fold: Dict[int, int] = {}
    for fold_id, (_, val_idx) in enumerate(kf.split(gly_in_pairs)):
        for j in val_idx:
            g_row = gly_in_pairs[j]
            row2fold[g_row] = int(fold_id)

    if len(row2fold) != len(gly_in_pairs):
        raise RuntimeError("KFold assignment failed: some glycans did not receive a fold id.")

    return row2fold, gly_in_pairs


# ----------------------------
# 4) Label structures
# ----------------------------
def build_g2true(
    pairs: List[Tuple[int, int]],
) -> Dict[int, Set[int]]:
    g2true: Dict[int, Set[int]] = defaultdict(set)
    for g, m in pairs:
        g2true[g].add(m)
    return g2true


# ----------------------------
# 5) Global frequency baseline
# ----------------------------
def build_global_mesh_frequency(
    train_pairs: List[Tuple[int, int]],
) -> Dict[int, int]:
    """
    Count MeSH frequency in train pairs.
    """
    counter = Counter()
    for _, m in train_pairs:
        counter[m] += 1
    return dict(counter)


def rank_mesh_by_global_frequency(
    mesh_freq: Dict[int, int],
) -> List[int]:
    """
    Rank MeSH labels by descending train frequency.
    """
    return [m for m, _ in sorted(mesh_freq.items(), key=lambda x: (-x[1], x[0]))]


def eval_global_frequency_baseline(
    *,
    train_pairs: List[Tuple[int, int]],
    query_g2true: Dict[int, Set[int]],
    topks: List[int] | Tuple[int, ...] = (1, 10, 15, 20, 25, 30),
) -> Dict[str, float]:
    topks = sorted(set(int(k) for k in topks))
    max_k = max(topks)

    mesh_freq = build_global_mesh_frequency(train_pairs)
    ranked_all = rank_mesh_by_global_frequency(mesh_freq)[:max_k]

    hit = {k: 0 for k in topks}
    rec = {k: 0.0 for k in topks}
    prec = {k: 0.0 for k in topks}
    mrr = {k: 0.0 for k in topks}
    n = 0

    for _, true_set in query_g2true.items():
        true_set = set(true_set)
        denom_true = max(1, len(true_set))
        n += 1

        for k in topks:
            topk = ranked_all[:k]

            if any(m in true_set for m in topk):
                hit[k] += 1

            inter = len(set(topk).intersection(true_set))
            rec[k] += inter / denom_true
            prec[k] += inter / float(k)

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
# 6) DEV CV
# ----------------------------
def run_dev_cv(
    *,
    dev_pairs: List[Tuple[int, int]],
    row2fold_dev: Dict[int, int],
    n_folds: int,
    topks: List[int],
) -> pd.DataFrame:
    all_rows: List[Dict[str, float]] = []

    for fold in range(n_folds):
        train_pairs = [(g, m) for (g, m) in dev_pairs if row2fold_dev[g] != fold]
        val_pairs = [(g, m) for (g, m) in dev_pairs if row2fold_dev[g] == fold]

        if not train_pairs or not val_pairs:
            continue

        g2true_val = build_g2true(val_pairs)
        metrics = eval_global_frequency_baseline(
            train_pairs=train_pairs,
            query_g2true=g2true_val,
            topks=topks,
        )

        row = {
            "fold": float(fold),
            "n_train_pairs": float(len(train_pairs)),
            "n_val_pairs": float(len(val_pairs)),
            "n_val_glycans": float(len(g2true_val)),
        }
        row.update(metrics)
        all_rows.append(row)

    return pd.DataFrame(all_rows)


# ----------------------------
# 7) CLI
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")

    ap.add_argument("--study_dir", required=True)

    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--test_seed", type=int, default=123)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)

    ap.add_argument("--metric", default="recall@25")

    return ap


# ----------------------------
# 8) Main
# ----------------------------
def main():
    args = build_argparser().parse_args()

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    # load glycan IDs from embedding CSV
    gly_ids, gly_id2row = load_gly_ids_from_embedding_csv(args.gly_emb_csv)

    # build mesh vocab from pairs CSV
    mesh_ids, mesh_id2row = load_mesh_ids_from_pairs_csv(
        args.pairs_csv,
        mesh_list_col=args.mesh_list_col,
        mesh_sep=args.mesh_sep,
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

    print(
        f"[pairs] built={len(pairs)} "
        f"missing_gly={missing_gly} "
        f"missing_mesh={missing_mesh} "
        f"n_mesh_labels={len(mesh_ids)}"
    )

    # fixed TEST split
    test_glycans, dev_glycans = make_fixed_test_split(
        pairs=pairs,
        test_frac=float(args.test_frac),
        seed=int(args.test_seed),
    )
    dev_pairs = filter_pairs_by_glycans(pairs, dev_glycans)
    test_pairs = filter_pairs_by_glycans(pairs, test_glycans)

    if not dev_pairs or not test_pairs:
        raise RuntimeError(f"Split produced empty DEV or TEST. dev_pairs={len(dev_pairs)} test_pairs={len(test_pairs)}")

    # fixed DEV fold assignment
    row2fold_dev, gly_in_dev_pairs = make_kfold_assignment_on_unique_glycans(
        pairs=dev_pairs,
        n_folds=int(args.n_folds),
        seed=int(args.cv_seed),
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
    print(f"[split] wrote DEV fold assignment: {assign_path.as_posix()} (cv_seed={args.cv_seed})")

    split_out = {
        "test_frac": float(args.test_frac),
        "test_seed": int(args.test_seed),
        "cv_seed": int(args.cv_seed),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_dev": int(len(dev_pairs)),
        "n_pairs_test": int(len(test_pairs)),
        "n_unique_glycans_total": int(len({g for (g, _) in pairs})),
        "n_unique_glycans_dev": int(len(dev_glycans)),
        "n_unique_glycans_test": int(len(test_glycans)),
        "n_mesh_labels": int(len(mesh_ids)),
        "test_glycan_rows": sorted(int(x) for x in test_glycans),
    }
    (study_dir / "split_test_glycans.json").write_text(json.dumps(split_out, indent=2), encoding="utf-8")
    print(f"[split] DEV pairs={len(dev_pairs)} TEST pairs={len(test_pairs)} "
          f"DEV glycans={len(dev_glycans)} TEST glycans={len(test_glycans)}")

    # metrics
    topks = [1, 10, 15, 20, 25, 30]
    k = _parse_metric_k(args.metric)
    if k is not None and k not in topks:
        topks.append(k)
    topks = sorted(set(topks))

    # DEV CV
    dev_cv_df = run_dev_cv(
        dev_pairs=dev_pairs,
        row2fold_dev=row2fold_dev,
        n_folds=int(args.n_folds),
        topks=topks,
    )
    if dev_cv_df.empty:
        raise RuntimeError("No DEV CV results were produced.")

    dev_cv_summary_path = study_dir / "dev_cv_summary.csv"
    dev_cv_df.to_csv(dev_cv_summary_path, index=False)
    print(f"[dev-cv] wrote: {dev_cv_summary_path.as_posix()}")

    mean_metrics = {}
    for col in dev_cv_df.columns:
        if col == "fold":
            continue
        try:
            mean_metrics[col] = float(dev_cv_df[col].astype(float).mean())
        except Exception:
            pass

    print(f"[dev-cv] mean {args.metric} = {mean_metrics.get(args.metric, float('nan')):.6f}")

    # Final TEST evaluation using all DEV as train prior
    g2true_test = build_g2true(test_pairs)
    test_metrics = eval_global_frequency_baseline(
        train_pairs=dev_pairs,
        query_g2true=g2true_test,
        topks=topks,
    )

    test_out = {
        "method": "global_cooccurrence_frequency",
        "objective_metric": str(args.metric),
        "dev_cv_mean_metrics": mean_metrics,
        "fixed_test": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
            "n_pairs_test": int(len(test_pairs)),
            "n_unique_glycans_test": int(len(test_glycans)),
        },
        "dev_reference": {
            "n_pairs_dev": int(len(dev_pairs)),
            "n_unique_glycans_dev": int(len(dev_glycans)),
        },
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }

    final_path = study_dir / "final_test_metrics.json"
    final_path.write_text(json.dumps(test_out, indent=2), encoding="utf-8")

    print(f"[final] wrote: {final_path.as_posix()}")
    print("[final] TEST metric summary:")
    keys_show = [f"hit@{k}" for k in topks] + [f"recall@{k}" for k in topks] + [f"mrr@{k}" for k in topks]
    keys_show = [k for k in keys_show if k in test_metrics]
    for k in keys_show:
        print(f"  {k}: {test_metrics[k]:.6f}")


if __name__ == "__main__":
    main()