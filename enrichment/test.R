rm(list = ls())
options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(dplyr)
  library(clusterProfiler)
  library(plotly)
})

setwd("./enrichment")

# =========================
# 0) Inputs
# =========================
gmt_file <- "mesh_descriptor_all.gmt"  

# =========================
# 1) Load GMT and derive glycan universe from the GMT itself
# =========================
gset <- read.gmt(gmt_file)  # columns: term, gene
stopifnot(all(c("term","gene") %in% colnames(gset)))

all_glycans <- unique(gset$gene)
length(all_glycans)

# =========================
# 2) Dummy DEG-like table using ONLY glycans present in the GMT
# =========================
set.seed(1)

n_test <- min(1000, length(all_glycans))  # Dummy number of glycans to analyze
deg.table <- tibble(
  gly_id = sample(all_glycans, n_test, replace = FALSE),
  log2FC = rnorm(n_test, mean = 0, sd = 0.4),
  padj  = runif(n_test),
  comp  = sample(c("cluster1_vs_non_resilience", "cluster2_vs_non_resilience"),
                 n_test, replace = TRUE)
) %>%
  mutate(direction = case_when(
    comp == "cluster1_vs_non_resilience" & log2FC >  0.1 & padj < 0.05 ~ "c1_high",
    comp == "cluster1_vs_non_resilience" & log2FC < -0.1 & padj < 0.05 ~ "c1_low",
    comp == "cluster2_vs_non_resilience" & log2FC >  0.1 & padj < 0.05 ~ "c2_high",
    comp == "cluster2_vs_non_resilience" & log2FC < -0.1 & padj < 0.05 ~ "c2_low",
    TRUE ~ ""
  ))

table(deg.table$direction)
universe_gly <- unique(deg.table$gly_id)

# =========================
# 3) Enrichment (ORA) per direction
# =========================
dr.list <- sort(unique(deg.table$direction))
dr.list <- dr.list[dr.list != ""]

all_res <- list()

for (dr in dr.list) {
  query_gly <- deg.table %>%
    filter(direction == dr) %>%
    pull(gly_id) %>%
    unique()
  
  if (length(query_gly) == 0) next
  
  egmt <- tryCatch(
    enricher(
      gene          = query_gly,
      TERM2GENE     = gset,
      universe      = universe_gly,
      pvalueCutoff  = 1.0,
      qvalueCutoff  = 1.0
    ),
    error = function(e) NULL
  )
  
  if (is.null(egmt) || nrow(egmt@result) == 0) next
  
  eres <- as.data.frame(egmt@result) %>%
    mutate(direction = dr)
  
  all_res[[dr]] <- eres
}

all_eres <- bind_rows(all_res)
all_eres_sig <- all_eres %>% filter(pvalue < 0.05)

# =========================
# 4) Outputs
# =========================
cat("all_eres:", dim(all_eres)[1], "rows x", dim(all_eres)[2], "cols\n")
cat("all_eres_sig (p<0.05):", dim(all_eres_sig)[1], "rows x", dim(all_eres_sig)[2], "cols\n")

head(all_eres, 10)
# head(all_eres_sig, 10)

# write.csv(all_eres, "EnrichmentTable_glycan_MeSH_dummy.csv", row.names = FALSE)
# write.csv(all_eres_sig, "EnrichmentTable_glycan_MeSH_dummy_p005.csv", row.names = FALSE)

# =========================
#  5) Plotly dot plot for one direction
# =========================
dr_pick <- "c1_high"   

dfp <- all_eres_sig %>%
  filter(direction == dr_pick) %>%
  mutate(
    GeneRatioNum = sapply(GeneRatio, function(x) {
      v <- strsplit(x, "/")[[1]]
      as.numeric(v[1]) / as.numeric(v[2])
    }),
    neglog10_FDR = -log10(p.adjust + 1e-300)
  ) %>%
  arrange(desc(GeneRatioNum)) %>%   
  slice_head(n = 20) %>%            # Top20
  mutate(
    Description = factor(
      Description,
      levels = rev(Description)     # Reverse so top-ranked items appear at the top of the plot
    )
  )

p <- plot_ly(
  data = dfp,
  x = ~GeneRatioNum,
  y = ~Description,
  type = "scatter",
  mode = "markers",
  marker = list(
    sizemode = "area",
    size = ~GeneRatioNum,
    color = ~neglog10_FDR,
    showscale = TRUE,
    sizeref = (max(dfp$GeneRatioNum, na.rm = TRUE) / 40)^2
  ),
  text = ~paste0(
    "Term: ", Description,
    "<br>GeneRatio: ", GeneRatio,
    "<br>Count: ", Count,
    "<br>FDR: ", signif(p.adjust, 3)
  ),
  hoverinfo = "text"
) %>%
  layout(
    title = paste0("Top20 Enrichment (by GeneRatio): ", dr_pick),
    xaxis = list(title = "GeneRatio"),
    yaxis = list(title = ""),
    margin = list(l = 300)
  )

p

# =========================
# 4) Outputs by parents
# =========================
parent <- read.csv("mesh_with_parent.csv")


