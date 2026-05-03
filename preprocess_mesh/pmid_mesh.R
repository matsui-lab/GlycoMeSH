library(data.table)

df <- readRDS("data/raw/glycan/pmid_mesh_long.rds")
write.csv(df, "data/raw/glycan/pmid_mesh_long.csv", quote=F)
