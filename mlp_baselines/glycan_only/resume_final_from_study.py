#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
resume_final_from_study.py
==========================

Resume ONLY the final stage from an already-finished Optuna study:
  1) load existing study.db
  2) get best trial params
  3) resolve fixed params (e.g. force_pu_loss => loss_type='pu_weighted')
  4) rebuild fixed DEV/TEST split
  5) retrain on ALL DEV
  6) evaluate on fixed TEST

This avoids rerunning Optuna trials.

Example:
python resume_final_from_study.py \
  --base_script ./glycan_only/optuna_glycan_mlp_holdout_test_cv5_pu_weighted.py \
  --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
  --study_dir ./data/analysis/multilabel_glycan_only/optuna_holdout_cv5_pu_weighted_run01 \
  --study_name glycan_mlp_multilabel_holdout_test_cv5_pu_weighted_run01 \
  --metric recall@25 \
  --test_frac 0.15 \
  --test_seed 123 \
  --epochs_stage1 50 \
  --epochs_stage2 100 \
  --stage2_init best \
  --grad_clip 1.0 \
  --num_workers 0 \
  --eval_every 1 \
  --eval_batch 2048 \
  --best_k 25 \
  --best_metric auto \
  --es_metric recall@25 \
  --patience_stage1 5 \
  --patience_stage2 10 \
  --min_delta 1e-3 \
  --warmup_epochs 2 \
  --device cuda \
  --seed 42 \
  --n_folds 5 \
  --force_pu_loss
"""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
from typing import Any, Dict

import optuna


def load_module_from_path(module_path: str):
    import sys

    module_path = str(Path(module_path).resolve())
    module_name = "base_train_module"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {module_path}")

    mod = importlib.util.module_from_spec(spec)

    # Workaround for Python 3.12 dataclass behavior:
    # register the module in sys.modules before exec_module().
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)
    return mod


def resolve_final_hparams(best_params: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Optuna best_params only contains tuned params.
    Fill fixed params that may be missing, especially loss_type when --force_pu_loss is used.
    """
    bp = dict(best_params)

    if getattr(args, "force_pu_loss", False):
        bp["loss_type"] = "pu_weighted"

    if "loss_type" not in bp:
        bp["loss_type"] = "bce"

    if "pu_pos_coef" not in bp:
        bp["pu_pos_coef"] = 1.0

    if "pu_unlabeled_coef" not in bp:
        bp["pu_unlabeled_coef"] = 1.0 if bp["loss_type"] == "bce" else 0.05

    return bp


def build_argparser():
    ap = argparse.ArgumentParser()

    # path to the ORIGINAL training script
    ap.add_argument("--base_script", required=True,
                    help="Path to optuna_glycan_mlp_holdout_test_cv5_pu_weighted.py")

    # same data args as original
    ap.add_argument("--gly_emb_csv", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--gly_id_col", default="glytoucan_ac")
    ap.add_argument("--mesh_list_col", default="descriptor_ui_list")
    ap.add_argument("--mesh_sep", default=";")
    ap.add_argument("--no_l2_norm", action="store_true")
    ap.add_argument("--mesh_emb_csv", required=True)

    # fixed split args
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--test_seed", type=int, default=123)

    # study
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--study_name", required=True)
    ap.add_argument("--storage", default="")

    # metric / schedule / training
    ap.add_argument("--metric", default="recall@25")
    ap.add_argument("--best_metric", default="auto")
    ap.add_argument("--best_k", type=int, default=25)

    ap.add_argument("--epochs_stage1", type=int, default=50)
    ap.add_argument("--epochs_stage2", type=int, default=100)
    ap.add_argument("--stage2_init", type=str, default="best", choices=["best", "last"])

    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--es_metric", default="recall@25")
    ap.add_argument("--patience_stage1", type=int, default=5)
    ap.add_argument("--patience_stage2", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # loss
    ap.add_argument("--force_pu_loss", action="store_true")

    # behavior
    ap.add_argument("--output_subdir", default="final_dev_train_resume",
                    help="Output subdir under study_dir")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing test_metrics.json if present")

    return ap


def main():
    args = build_argparser().parse_args()

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    out_dir = study_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "test_metrics.json"
    if out_json.exists() and (not args.overwrite):
        raise FileExistsError(
            f"{out_json} already exists. Use --overwrite to replace it."
        )

    base = load_module_from_path(args.base_script)

    storage = args.storage.strip() if args.storage.strip() else f"sqlite:///{(study_dir / 'study.db').as_posix()}"

    print("[resume] loading study")
    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage,
    )

    best_trial = study.best_trial
    best_params_raw = dict(best_trial.params)
    best_params = resolve_final_hparams(best_params_raw, args)

    print(f"[resume] best trial number: {best_trial.number}")
    print(f"[resume] best value: {study.best_value}")
    print(f"[resume] raw best params: {best_params_raw}")
    print(f"[resume] resolved final params: {best_params}")

    # load data using original helpers
    G, gly_ids, gly_id2row = base.load_gly_embeddings_from_csv(
        args.gly_emb_csv,
        normalize=(not args.no_l2_norm),
    )
    mesh_ids, mesh_id2row = base.load_mesh_ids_from_embedding_csv(args.mesh_emb_csv)

    pairs, missing_gly, missing_mesh = base.build_pairs_from_csv_with_given_mesh_vocab(
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
        f"n_mesh_labels(full_candidate)={len(mesh_ids)}"
    )

    Y, _ = base.build_multilabel_targets(
        pairs=pairs,
        n_gly=G.size(0),
        n_mesh=len(mesh_ids),
    )

    # rebuild SAME fixed split as original
    test_glycans, dev_glycans = base.make_fixed_test_split(
        pairs=pairs,
        test_frac=float(args.test_frac),
        seed=int(args.test_seed),
    )
    dev_pairs = base.filter_pairs_by_glycans(pairs, dev_glycans)
    test_pairs = base.filter_pairs_by_glycans(pairs, test_glycans)

    if not dev_pairs or not test_pairs:
        raise RuntimeError(
            f"Split produced empty DEV or TEST. dev_pairs={len(dev_pairs)} test_pairs={len(test_pairs)}"
        )

    final_cfg = base.TrainCfg(
        hidden1=int(best_params["hidden1"]),
        hidden2=int(best_params["hidden2"]),
        dropout=float(best_params["dropout"]),

        lr_stage1=float(best_params["lr_stage1"]),
        lr_stage2=float(best_params["lr_stage2"]),
        weight_decay=float(best_params["weight_decay"]),
        epochs_stage1=int(args.epochs_stage1),
        epochs_stage2=int(args.epochs_stage2),
        stage2_init=str(args.stage2_init),
        grad_clip=float(args.grad_clip),

        batch_size=int(best_params["batch_size"]),
        num_workers=int(args.num_workers),

        eval_every=int(args.eval_every),
        eval_batch=int(args.eval_batch),
        best_k=int(args.best_k),
        best_metric=str(args.best_metric if args.best_metric != "auto" else args.metric),
        es_metric=str(args.es_metric),
        early_stop=False,   # final train is no-val
        patience_stage1=int(args.patience_stage1),
        patience_stage2=int(args.patience_stage2),
        min_delta=float(args.min_delta),
        warmup_epochs=int(args.warmup_epochs),

        pos_weight_mode=str(best_params["pos_weight_mode"]),
        pos_weight_clip_max=float(best_params["pos_weight_clip_max"]),

        loss_type=str(best_params["loss_type"]),
        pu_pos_coef=float(best_params["pu_pos_coef"]),
        pu_unlabeled_coef=float(best_params["pu_unlabeled_coef"]),

        seed=int(args.seed),
        device=str(args.device),
        n_folds=int(args.n_folds),
    )

    device = base._get_device(final_cfg.device)

    print("[resume] start final retrain on ALL DEV")
    model = base._train_two_stage_no_val(
        cfg=final_cfg,
        G=G,
        Y=Y,
        dev_pairs=dev_pairs,
        out_dir=out_dir,
        device=device,
    )

    eval_topks = [1, 10, 15, 20, 25, 30]
    k = base._parse_metric_k(args.metric)
    if k is not None and k not in eval_topks:
        eval_topks.append(k)
    eval_topks = sorted(set(eval_topks))

    print("[resume] evaluate on fixed TEST")
    test_metrics = base._eval_on_test(
        model=model,
        G=G,
        test_pairs=test_pairs,
        device=device,
        topks=eval_topks,
    )

    out = {
        "resume_mode": "final_only_from_existing_study",
        "base_script": str(Path(args.base_script).resolve()),
        "study_name": args.study_name,
        "storage": storage,
        "best_trial_number": int(best_trial.number),
        "best_value_dev_cv_mean": float(study.best_value),
        "best_params_raw_from_study": best_params_raw,
        "best_params_resolved_for_final": best_params,
        "objective_metric": str(args.metric),
        "final_dev_train": {
            "epochs_stage1": int(args.epochs_stage1),
            "epochs_stage2": int(args.epochs_stage2),
            "batch_size": int(final_cfg.batch_size),
            "seed": int(final_cfg.seed),
            "n_mesh_labels": int(len(mesh_ids)),
        },
        "fixed_test": {
            "test_frac": float(args.test_frac),
            "test_seed": int(args.test_seed),
            "n_pairs_test": int(len(test_pairs)),
            "n_unique_glycans_test": int(len(test_glycans)),
        },
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "mesh_emb_csv": str(args.mesh_emb_csv),
    }

    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[resume] wrote: {out_json.as_posix()}")

    print("[resume] TEST metric summary:")
    keys_show = (
        [f"hit@{k}" for k in eval_topks]
        + [f"recall@{k}" for k in eval_topks]
        + [f"mrr@{k}" for k in eval_topks]
    )
    keys_show = [k for k in keys_show if k in test_metrics]
    for k in keys_show:
        print(f"  {k}: {test_metrics[k]:.6f}")

    print("[resume] done.")


if __name__ == "__main__":
    main()
    
# python resume_final_from_study.py \
#   --base_script ./glycan_only/optuna_glycan_mlp_holdout_test_cv5_pu_weighted.py \
#   --gly_emb_csv ./data/glycangt_embedding_2026/embeddings_large.csv \
#   --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
#   --pairs_csv ./data/glycan/glytoucan_iupac_mesh_filtered.csv \
#   --study_dir ./data/analysis/multilabel_glycan_only/optuna_holdout_cv5_pu_weighted_run01 \
#   --study_name glycan_mlp_multilabel_holdout_test_cv5_pu_weighted_run01 \
#   --metric recall@25 \
#   --test_frac 0.15 \
#   --test_seed 123 \
#   --epochs_stage1 50 \
#   --epochs_stage2 100 \
#   --stage2_init best \
#   --grad_clip 1.0 \
#   --num_workers 0 \
#   --eval_every 1 \
#   --eval_batch 2048 \
#   --best_k 25 \
#   --best_metric auto \
#   --es_metric recall@25 \
#   --patience_stage1 5 \
#   --patience_stage2 10 \
#   --min_delta 1e-3 \
#   --warmup_epochs 2 \
#   --device cuda \
#   --seed 42 \
#   --n_folds 5 \
#   --force_pu_loss