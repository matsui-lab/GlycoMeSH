#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """Sample from log-uniform distribution."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    # ---- fixed paths ----
    train_py = "./train_cl/train_cross_modal_stablerep_normal_cv.py"
    gly_emb = "./data/glycangt_embedding_2026/embeddings_large.csv"
    pairs_csv = "./data/glycan/glytoucan_iupac_mesh_filtered.csv"
    out_root = Path("./data/analysis/contrastive_learning/5_n_cv_random_search_filtered")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- mesh embedding candidates (4 types) ----
    mesh_emb_map = {
        "biobert": "./data/mesh/embedding/biobert_description_cls_filtered.csv",
        "medcpt": "./data/mesh/embedding/medcpt_description_cls_filtered.csv",
        "pubmedbert": "./data/mesh/embedding/pubmedbert_description_meanpool_filtered.csv",
        "sapbert": "./data/mesh/embedding/sapbert_name_cls_filtered.csv",
    }

    # ---- how many random HP samples per embedding ----
    n_trials_per_embedding = 50

    # ---- base seed (controls reproducibility) ----
    base_seed = 42

    for emb_name, mesh_emb in mesh_emb_map.items():
        emb_root = out_root / emb_name
        emb_root.mkdir(parents=True, exist_ok=True)

        for t in range(n_trials_per_embedding):
            # Per-trial RNG for reproducibility: same base_seed => same HP for same (emb_name, t)
            trial_seed = (base_seed * 1000003 + hash(emb_name) + t) & 0xFFFFFFFF
            rng = random.Random(trial_seed)

            hp = {
                "mesh_embedding": emb_name,
                "mesh_emb_csv": mesh_emb,
                "trial_seed": trial_seed,

                "lr_stage1": log_uniform(rng, 3e-4, 3e-3),
                "lr_stage2": log_uniform(rng, 3e-5, 3e-4),
                "tau": log_uniform(rng, 0.05, 0.3),
                "dropout": rng.uniform(0.0, 0.5),
                "weight_decay": log_uniform(rng, 1e-6, 1e-3),

                "pos_per_glycan": rng.choice([4, 8, 12, 16]),
                "batch_glycans": rng.choice([256, 512, 1024]),
                "out_dim": rng.choice([256, 512]),
                "hidden": rng.choice([512, 1024, 2048]),
            }

            trial_dir = emb_root / f"trial_{t:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "hp.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")

            cmd = [
                "python", train_py,
                "--gly_emb_csv", gly_emb,
                "--mesh_emb_csv", mesh_emb,
                "--pairs_csv", pairs_csv,
                "--out_dir", str(trial_dir),

                "--lr_stage1", f"{hp['lr_stage1']}",
                "--lr_stage2", f"{hp['lr_stage2']}",
                "--tau", f"{hp['tau']}",
                "--dropout", f"{hp['dropout']}",
                "--weight_decay", f"{hp['weight_decay']}",
                "--pos_per_glycan", str(hp["pos_per_glycan"]),
                "--batch_glycans", str(hp["batch_glycans"]),
                "--out_dim", str(hp["out_dim"]),
                "--hidden", str(hp["hidden"]),

                "--best_metric", "recall@25",
                "--es_metric", "recall@25",
                "--early_stop",
            ]

            run(cmd)


if __name__ == "__main__":
    main()
