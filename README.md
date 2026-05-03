# GlycoMeSH-BERT and Glycan Ontology

Source code accompanying the manuscript
**"Systematic functional annotation and enrichment analysis of glycans using multimodal contrastive learning."**

GlycoMeSH-BERT is a multimodal contrastive learning framework that aligns
glycans with Medical Subject Headings (MeSH) terms in a shared embedding
space. The trained model is used to construct the **Glycan Ontology**, a
non-redundant resource of glycan–MeSH associations, and to perform
glycan-centric enrichment analysis on glycomics and glycoproteomics data.

## Repository layout

```
.
├── preprocess_mesh/        # MeSH XML parsing, PMID -> MeSH term mapping
├── mesh_embed/             # MeSH text embeddings (BioBERT/PubMedBERT/MedCPT/SapBERT)
├── train_cl/               # GlycoMeSH-BERT (StableRep contrastive learning)
│                           # Includes baselines (cosine, co-occurrence) and inference scripts.
├── mlp_baselines/
│   ├── glycan_only/        # Multi-label MLP using glycan embeddings only
│   └── glycan_mesh/        # Dual MLP using glycan + MeSH embeddings
├── case_study/             # Application notes for OSCC and GPST000502 datasets
├── enrichment/             # clusterProfiler-based glycan enrichment example
├── shiny/                  # Web GUI tool (R Shiny) for Glycan Ontology enrichment
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── .zenodo.json            # Zenodo deposit metadata
```

The scripts assume a top-level `data/` directory holding inputs and
intermediate outputs (paths normalized to `./data/...`). Layout used in the
manuscript:

```
data/
├── raw/
│   ├── desc2026.xml                        # MeSH descriptors (NLM 2026 release)
│   ├── qual2026.xml                        # MeSH qualifiers (NLM 2026 release)
│   └── glycan/                             # GlyCosmos exports (PMIDs, structures)
├── mesh/
│   ├── mesh_descriptors.csv
│   ├── mesh_descriptors_withTree.csv
│   ├── mesh_descriptors_filtered.csv
│   └── embedding/{biobert,pubmedbert,medcpt,sapbert}_*.csv
├── glycan/
│   ├── glytoucan_iupac_mesh.csv
│   ├── glytoucan_iupac_mesh_filtered.csv
│   └── graph_id_cluster5_5cv.csv
├── glycangt_embedding_2026/
│   └── embeddings_large.csv                # Glycan embeddings from GlycanGT
└── analysis/                               # Hyperparameter searches and final models
```

The raw inputs are not bundled with this repository; they should be
obtained from the original sources (see "Data sources" below).

## Data sources

- **MeSH** descriptors and qualifiers: NLM 2026 release
  (https://www.nlm.nih.gov/mesh/).
- **Glycans**: GlyCosmos (https://glycosmos.org/) for IUPAC-condensed
  representations and glycan–PMID mappings; GlyTouCan accessions.
- **GlycanGT** embeddings: pretrained model from Kitani *et al.* (2026),
  *Bioinformatics*. Run the published GlycanGT inference pipeline to
  obtain `embeddings_large.csv`.
- **Application datasets**: macrophage glycomics (Dipta *et al.* 2024),
  OSCC glycomics (Carnielli *et al.* 2023), human AD glycoproteomics
  (Suttapitugsakul *et al.* 2022), mouse APP/PS1 glycoproteomics
  (Fang *et al.* 2025).

## Reproducing the analysis

The end-to-end pipeline consists of the following stages. Each script
exposes its inputs and outputs as CLI arguments; example invocations are
included in the docstring or trailing comment of each script.

1. **MeSH preprocessing** (`preprocess_mesh/`)
   - `xml_parser.py`, `xml_parser_qualifier.py`, `mesh_parser_withTree.py`
     parse `desc2026.xml` / `qual2026.xml` into descriptor/qualifier
     tables under `data/mesh/`.
   - `pubmed_to_mesh.R` and `pmid_mesh.R` build PMID → MeSH-term
     mappings used to construct glycan–MeSH annotations.

2. **MeSH embeddings** (`mesh_embed/mesh_embed.py`)
   - Encodes MeSH descriptor texts with BioBERT, PubMedBERT, MedCPT,
     and SapBERT. Outputs CSVs under `data/mesh/embedding/`.

3. **Contrastive learning** (`train_cl/`)
   - `hp_n_cv_rand_search_1st.py` — first-stage random search across
     MeSH embedding choices.
   - `optuna_sapbert_stablerep_holdout_test_cv5_wide.py` /
     `optuna_sapbert_stablerep_holdout.py` — Optuna Bayesian search
     with the StableRep multi-positive contrastive loss using the
     SapBERT MeSH embeddings.
   - `train_cross_modal_stablerep_normal_cv.py` /
     `train_cross_modal_stablerep_predefine_folds.py` — 5-fold CV
     training under fixed hyperparameters.
   - `train_cross_modal_stablerep_full_train.py` — final model trained
     on the full development set without a validation split (used for
     Glycan Ontology construction).
   - `infer_all_topk_from_final_model.py`,
     `infer_all_topk_from_optuna_2nd_Model.py` — top-k MeSH inference
     for all glycans.
   - `baseline_cosine_retrieval_holdout_test_cv5.py`,
     `global_cooccurrence_frequency_holdout_test_cv5.py` —
     non-contrastive baselines (cosine retrieval and co-occurrence).
   - `utils.py`, `utils_kfold.py` — shared training utilities.

4. **MLP baselines** (`mlp_baselines/`)
   - `glycan_only/` — multi-label MLP over the full MeSH vocabulary
     using glycan embeddings as input.
   - `glycan_mesh/` — dual MLP that ingests both glycan and SapBERT
     MeSH embeddings.
   - Each subdirectory provides Optuna hyperparameter search,
     full-train, inference, and resume scripts.

5. **Enrichment analysis** (`enrichment/test.R`)
   - Minimal example of `clusterProfiler::enricher` over the Glycan
     Ontology GMT.

6. **Web GUI** (`shiny/`)
   - `ui.R`, `server.R` define the R Shiny application; the GMT and
     parent-term mapping required to run the app are bundled here
     (`mesh_descriptor_all.gmt`, `mesh_with_parent.csv`).

## Software environment

Python 3.12.11 with the libraries listed in `requirements.txt`
(PyTorch 2.5.1+cu121, Optuna 4.7.0, scikit-learn 1.7.0, SciPy 1.15.3,
transformers, pandas, numpy).

R 4.4.0 with `clusterProfiler 4.14.6`, `dplyr`, `tibble`, `purrr`,
`stringr`, `xml2`, `httr2`, `shiny`, `plotly`. Install via
`BiocManager::install("clusterProfiler")` and `install.packages(...)`.

GPU is recommended for training; inference can be run on CPU.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite the accompanying manuscript.
