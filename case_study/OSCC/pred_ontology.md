# GlycoFunc

## GlycanGT  embedding
```bash
conda activate tokengt
cd ./glycoGT/
python get_embeddings.py --input_dir ./case_study/OSCC/out/emb --output_dir ./case_study/OSCC/out/glycangt/ --iupac_col top1_pred #--id_col new_id
```

## Infer topk prediction
### CandyCrunch returns structures as GlyTouCan IDs, whereas GlycoMeSH-BERT retrieves data from GlyCosmos. Although GlyCosmos is expected to integrate data from GlyTouCan, some entries are not consistent between the two.
### First, divide the CandyCrunch results into those with assigned IDs and those without; then perform inference for the entries lacking IDs.

```bash
cd ./glycan_only
python infer_all_topk_from_full_glycan_mlp.py \
  --study_dir ./data/analysis/multilabel_glycan_only/final_full_model_fullmesh \
  --gly_emb_csv ./case_study/OSCC/old/out/glycangt/embeddings_large.csv \
  --mesh_emb_csv ./data/mesh/embedding/sapbert_name_cls_filtered.csv \
  --topk 5000 \
  --chunk_g 2048 \
  --device cuda \
  --out_csv ./case_study/OSCC/out/mesh_pred/infer_all_glycan_topk.csv \
```

# GSEA analysis
## Because the number of glycans showing significant differences is small, threshold-based analysis reduces detection sensitivity (e.g., pathways may be identified as significant with only a single hit glycan). Therefore, GSEA was performed.