#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_embed.py
---------------------
Create MeSH-term embeddings from a dataframe (index = MeSH TERM ID, columns include: descriptor_name, label_text).

Models (4):
1) SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)  -> use df["descriptor_name"], CLS (last layer)
2) MedCPT Article Encoder (ncbi/MedCPT-Article-Encoder)     -> use df["label_text"], CLS (last layer)
   - MedCPT expects article input [title, abstract]; this script uses ["", label_text]
3) NeuML PubMedBERT embeddings (neuml/pubmedbert-base-embeddings) -> use df["label_text"], mean pooling
4) BioBERT (dmis-lab/biobert-base-cased-v1.2)               -> use df["label_text"], CLS (last layer)

Outputs:
- Separate CSV per model, each indexed by MeSH TERM ID (same as input df index)
- Columns: emb_000 ... emb_767

Example:
python mesh_embed.py \
  --input data/mesh_terms.csv \
  --filetype csv \
  --outdir out/embeddings \
  --device auto \
  --batch_size 128 \
  --cache_dir /path/to/hf_cache
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Utilities
# -----------------------------
def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_df(path: str, filetype: str, index_col: Optional[int] = 0) -> pd.DataFrame:
    if filetype == "csv":
        df = pd.read_csv(path, index_col=index_col)
    elif filetype == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    if "descriptor_name" not in df.columns or "label_text" not in df.columns:
        raise ValueError("Input dataframe must have columns: 'descriptor_name' and 'label_text'")

    # Keep index as MeSH TERM ID
    df.index = df.index.astype(str)
    return df


def ensure_list_str(series: pd.Series) -> List[str]:
    return series.fillna("").astype(str).tolist()


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: [B, T, H]
    attention_mask:   [B, T]
    returns:          [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = torch.sum(last_hidden_state * mask, dim=1)             # [B,H]
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)                  # [B,1]
    return summed / denom


def save_embeddings_csv(out_path: str, ids: Sequence[str], embs: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dim = embs.shape[1]
    cols = [f"emb_{i:03d}" for i in range(dim)]
    df_out = pd.DataFrame(embs, index=pd.Index(ids, name="mesh_id"), columns=cols)
    df_out.to_csv(out_path, index=True)


# -----------------------------
# HF Loading
# -----------------------------
@dataclass
class HFLoadArgs:
    model_id_or_path: str
    cache_dir: Optional[str]
    local_files_only: bool
    trust_remote_code: bool


def load_tokenizer_model(load_args: HFLoadArgs, device: torch.device, use_safetensors: bool = False) -> Tuple[AutoTokenizer, AutoModel]:
    tok = AutoTokenizer.from_pretrained(
        load_args.model_id_or_path,
        cache_dir=load_args.cache_dir,
        local_files_only=load_args.local_files_only,
        trust_remote_code=load_args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        load_args.model_id_or_path,
        cache_dir=load_args.cache_dir,
        local_files_only=load_args.local_files_only,
        trust_remote_code=load_args.trust_remote_code,
        use_safetensors=use_safetensors,
    ).to(device)
    model.eval()
    return tok, model

# -----------------------------
# Embedding functions
# -----------------------------
@torch.no_grad()
def embed_cls_single_text(
    texts: Sequence[str],
    tok: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
    use_fp16: bool,
) -> np.ndarray:
    all_embs: List[np.ndarray] = []

    autocast_enabled = (use_fp16 and device.type == "cuda")

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        enc = tok(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            out = model(**enc)
            # Standard: out.last_hidden_state [B,T,H]
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            cls = last_hidden[:, 0, :]  # [B,H]

        all_embs.append(cls.detach().float().cpu().numpy().astype(np.float32))

    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def embed_meanpool_single_text(
    texts: Sequence[str],
    tok: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
    use_fp16: bool,
) -> np.ndarray:
    all_embs: List[np.ndarray] = []

    autocast_enabled = (use_fp16 and device.type == "cuda")

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        enc = tok(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            out = model(**enc)
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            pooled = mean_pooling(last_hidden, enc["attention_mask"])  # [B,H]

        all_embs.append(pooled.detach().float().cpu().numpy().astype(np.float32))

    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def embed_medcpt_article_cls(
    descriptions: Sequence[str],
    tok: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
    use_fp16: bool,
) -> np.ndarray:
    """
    MedCPT Article Encoder expects inputs like: List[List[str]] (e.g., [title, abstract]).
    This script uses ["", label_text] to represent (title="", abstract=label_text).
    """
    all_embs: List[np.ndarray] = []

    autocast_enabled = (use_fp16 and device.type == "cuda")

    for i in tqdm(range(0, len(descriptions), batch_size), desc=desc):
        batch_desc = descriptions[i : i + batch_size]
        articles = [["", d] for d in batch_desc]

        enc = tok(
            articles,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            out = model(**enc)
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            cls = last_hidden[:, 0, :]  # [B,H]

        all_embs.append(cls.detach().float().cpu().numpy().astype(np.float32))

    return np.concatenate(all_embs, axis=0)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input dataframe path (csv/parquet). Index must be MeSH TERM ID.")
    ap.add_argument("--filetype", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--outdir", required=True, help="Directory to write per-model embedding CSVs.")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--use_fp16", action="store_true", help="Use autocast fp16 on CUDA for faster inference.")
    ap.add_argument("--cache_dir", default=None, help="Hugging Face cache directory (optional).")
    ap.add_argument("--local_files_only", action="store_true", help="Load models only from local cache/files (offline).")
    ap.add_argument("--use_safetensors", action="store_true", help="Force loading model weights from safetensors when available (avoids torch.load).",)

    # Max lengths per model
    ap.add_argument("--sapbert_max_length", type=int, default=25, help="SapBERT max_length (official example uses 25).")
    ap.add_argument("--medcpt_max_length", type=int, default=512)
    ap.add_argument("--pubmedbert_max_length", type=int, default=512)
    ap.add_argument("--biobert_max_length", type=int, default=512)

    # Model IDs / paths
    ap.add_argument("--sapbert_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--medcpt_model", default="ncbi/MedCPT-Article-Encoder")
    ap.add_argument("--pubmedbert_model", default="neuml/pubmedbert-base-embeddings")
    ap.add_argument("--biobert_model", default="dmis-lab/biobert-base-cased-v1.2")

    args = ap.parse_args()

    device = get_device(args.device)
    print(f"[INFO] device = {device}")
    print(f"[INFO] local_files_only = {args.local_files_only}")
    if args.cache_dir:
        print(f"[INFO] cache_dir = {args.cache_dir}")

    df = load_df(args.input, args.filetype)
    ids = df.index.astype(str).tolist()

    names = ensure_list_str(df["descriptor_name"])
    descs = ensure_list_str(df["label_text"])

    # Common load args
    def la(model_id_or_path: str, trust_remote_code: bool = False) -> HFLoadArgs:
        return HFLoadArgs(
            model_id_or_path=model_id_or_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            trust_remote_code=trust_remote_code,
        )

    # -------------------------
    # 1) SapBERT: name -> CLS
    # -------------------------
    tok_sap, model_sap = load_tokenizer_model(la(args.sapbert_model, trust_remote_code=False), device,
                                              use_safetensors=args.use_safetensors,)
    sap_embs = embed_cls_single_text(
        texts=names,
        tok=tok_sap,
        model=model_sap,
        device=device,
        batch_size=args.batch_size,
        max_length=args.sapbert_max_length,
        desc="SapBERT (CLS; name)",
        use_fp16=args.use_fp16,
    )
    save_embeddings_csv(os.path.join(args.outdir, "sapbert_name_cls.csv"), ids, sap_embs)

    # -------------------------
    # 2) MedCPT: label_text -> CLS (["", label_text])
    # -------------------------
    tok_med, model_med = load_tokenizer_model(la(args.medcpt_model, trust_remote_code=False), device,
                                              use_safetensors=args.use_safetensors,)
    med_embs = embed_medcpt_article_cls(
        descriptions=descs,
        tok=tok_med,
        model=model_med,
        device=device,
        batch_size=args.batch_size,
        max_length=args.medcpt_max_length,
        desc="MedCPT Article (CLS; label_text)",
        use_fp16=args.use_fp16,
    )
    save_embeddings_csv(os.path.join(args.outdir, "medcpt_description_cls.csv"), ids, med_embs)

    # -------------------------
    # 3) NeuML PubMedBERT embeddings: label_text -> mean pooling
    # -------------------------
    tok_pub, model_pub = load_tokenizer_model(la(args.pubmedbert_model, trust_remote_code=False), device,
                                              use_safetensors=args.use_safetensors,)
    pub_embs = embed_meanpool_single_text(
        texts=descs,
        tok=tok_pub,
        model=model_pub,
        device=device,
        batch_size=args.batch_size,
        max_length=args.pubmedbert_max_length,
        desc="NeuML PubMedBERT embeddings (meanpool; label_text)",
        use_fp16=args.use_fp16,
    )
    save_embeddings_csv(os.path.join(args.outdir, "pubmedbert_description_meanpool.csv"), ids, pub_embs)

    # -------------------------
    # 4) BioBERT: label_text -> CLS
    # -------------------------
    tok_bio, model_bio = load_tokenizer_model(la(args.biobert_model, trust_remote_code=False), device,
                                              use_safetensors=args.use_safetensors,)
    bio_embs = embed_cls_single_text(
        texts=descs,
        tok=tok_bio,
        model=model_bio,
        device=device,
        batch_size=args.batch_size,
        max_length=args.biobert_max_length,
        desc="BioBERT (CLS; label_text)",
        use_fp16=args.use_fp16,
    )
    save_embeddings_csv(os.path.join(args.outdir, "biobert_description_cls.csv"), ids, bio_embs)

    print("[DONE] Wrote embeddings:")
    print(f" - {os.path.join(args.outdir, 'sapbert_name_cls.csv')}")
    print(f" - {os.path.join(args.outdir, 'medcpt_description_cls.csv')}")
    print(f" - {os.path.join(args.outdir, 'pubmedbert_description_meanpool.csv')}")
    print(f" - {os.path.join(args.outdir, 'biobert_description_cls.csv')}")


if __name__ == "__main__":
    main()
