library(httr2)

# Set working directory to the repository root if needed.
# setwd("/path/to/repo")
pmid_test <- "28122943"

library(xml2)
library(dplyr)
library(purrr)
library(tibble)
library(stringr)
library(tidyr)

# glycan citation data
df <- read.csv("./data/raw/glycan/glycan_citations_glytoucan.csv")
names(df)
df <- df[,c("glytoucan_ac","xref_id")]
names(df) <- c("glytoucan_ac","pmid")

# glycan data with IUPAC condensed
glycan <- read.csv("./data/raw/glycan/glycosmos_glycans_list.csv")
glycan <- glycan[,c(1,2)]
names(glycan)
names(glycan) <- c("glytoucan_ac","iupac")

df_merged <- left_join(df, glycan, by = "glytoucan_ac")

df_pubmed <- df_merged %>%
  transmute(
    glytoucan_ac = as.character(glytoucan_ac),
    pmid = as.character(pmid),
    iupac = as.character(iupac)
  ) %>%
  filter(!is.na(pmid), str_detect(pmid, "^[0-9]+$"))

pmids <- sort(unique(df_pubmed$pmid))
length(pmids)

fetch_mesh_for_pmids <- function(pmids_batch,
                                 api_key = NULL,
                                 email = NULL,
                                 timeout_sec = 180,
                                 max_tries = 6,
                                 sleep_sec = 0.34) {
  ids <- paste(pmids_batch, collapse = ",")
  
  for (i in seq_len(max_tries)) {
    ok <- TRUE
    res <- tryCatch({
      req <- request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi") |>
        req_url_query(db="pubmed", id=ids, rettype="xml", retmode="xml") |>
        req_timeout(timeout_sec)
      
      if (!is.null(api_key)) req <- req |> req_url_query(api_key = api_key)
      if (!is.null(email))   req <- req |> req_url_query(email = email)
      
      resp <- req |> req_perform()
      stopifnot(resp_status(resp) == 200)
      
      xml_txt <- resp_body_string(resp)
      if (!str_detect(xml_txt, "^<")) {
        writeLines(substr(xml_txt, 1, 500), "bad_response_head.txt")
        stop("Non-XML response detected")
      }
      doc <- read_xml(xml_txt)
      articles <- xml_find_all(doc, ".//PubmedArticle")
      
      map_dfr(articles, function(a) {
        pmid_node <- xml_find_first(a, ".//MedlineCitation/PMID")
        pmid <- if (!is.na(pmid_node)) xml_text(pmid_node) else NA_character_
        
        desc <- xml_find_all(a, ".//MeshHeadingList/MeshHeading/DescriptorName")
        
        if (length(desc) == 0) {
          return(tibble(pmid = pmid, descriptor_ui = NA_character_, descriptor_name = NA_character_))
        }
        
        tibble(
          pmid = pmid,
          descriptor_ui = xml_attr(desc, "UI"),
          descriptor_name = xml_text(desc)
        ) %>% distinct()
      })
    }, error = function(e) {
      ok <<- FALSE
      e
    })
    
    if (ok) {
      Sys.sleep(sleep_sec)
      return(res)
    }
    
    # backoff
    Sys.sleep(min(60, 2^(i-1)))
  }
  
  stop("Failed to fetch batch after retries.")
}

cache_path <- "./data/raw/glycan/pmid_mesh_long.rds"
batch_size <- 100

if (file.exists(cache_path)) {
  message("Loading cache: ", cache_path)
  mesh_long <- readRDS(cache_path)
} else {
  batches <- split(pmids, ceiling(seq_along(pmids) / batch_size))
  
  mesh_long <- map_dfr(seq_along(batches), function(k) {
    message(sprintf("Batch %d/%d (n=%d)", k, length(batches), length(batches[[k]])))
    fetch_mesh_for_pmids(batches[[k]], api_key = "2fc25ecff6a27c94ca7e59a9e5a47940a509", email = "kitani.akihiro.b4@s.mail.nagoya-u.ac.jp")
  })
  
  saveRDS(mesh_long, cache_path)
  message("Saved cache: ", cache_path)
}

mesh_long %>% filter(!is.na(descriptor_ui)) %>% head(20)

covered_pmids <- mesh_long %>% distinct(pmid) %>% pull(pmid)
length(covered_pmids)
length(pmids)
setdiff(pmids, covered_pmids) %>% head(20)

df_with_mesh_long <- df_pubmed %>%
  left_join(mesh_long %>% distinct(pmid, descriptor_ui, descriptor_name), by="pmid")

df_with_mesh_long %>% head(20)

glycan_mesh <- df_with_mesh_long %>%
  filter(!is.na(descriptor_ui)) %>%
  group_by(glytoucan_ac) %>%
  summarise(
    iupac = dplyr::first(iupac),  
    descriptor_ui_list = paste(sort(unique(descriptor_ui)), collapse=";"),
    descriptor_name_list = paste(sort(unique(descriptor_name)), collapse=";"),
    .groups = "drop"
  )

glycan_mesh %>% head(20)
glycan_mesh <- glycan_mesh[,c(1,2,3)]

glycan_base <- df_pubmed %>%
  group_by(glytoucan_ac) %>%
  summarise(iupac = dplyr::first(iupac), .groups="drop")
length(unique(glycan_base$glytoucan_ac))

final <- glycan_base %>%
  left_join(glycan_mesh %>% select(glytoucan_ac, descriptor_ui_list),
            by="glytoucan_ac")

final %>% head(20)
final_filtered <- final %>%
  filter(
    !is.na(iupac),
    str_trim(iupac) != "",
    str_count(iupac, fixed("(")) >= 1,
    str_count(iupac, fixed(")")) >= 1
  )

covered_pmids <- mesh_long %>% distinct(pmid) %>% pull(pmid)
cat("pmids:", length(pmids), "covered_pmids:", length(covered_pmids), "\n")
cat("missing:", length(setdiff(pmids, covered_pmids)), "\n")
cat("mesh rows:", nrow(mesh_long), "\n")
cat("mesh non-NA:", sum(!is.na(mesh_long$descriptor_ui)), "\n")
cat("final:", nrow(final), "final_filtered:", nrow(final_filtered), "\n")



# Check for duplicates in records that were truncated at "("
df_norm <- final_filtered %>%
  mutate(
    n_lpar = str_count(iupac, fixed("(")),
    n_rpar = str_count(iupac, fixed(")")),
    paren_mismatch = n_lpar != n_rpar
  )

table(df_norm$paren_mismatch)

df_norm <- df_norm %>%
  mutate(
    normalized_iupac = ifelse(
      paren_mismatch,
      str_replace(iupac, "\\([^\\(]*$", ""),
      iupac
    ),
    normalized_iupac = str_trim(normalized_iupac)
  )

dup_stats <- df_norm %>%
  count(normalized_iupac, name = "n_variants") %>%
  arrange(desc(n_variants))

summary(dup_stats$n_variants)

dup_stats %>%
  filter(n_variants > 1) %>%
  head(20)

glycan_mesh_normalized <- df_norm %>%
  separate_rows(descriptor_ui_list, sep = ";") %>%
  filter(descriptor_ui_list != "", !is.na(descriptor_ui_list)) %>%
  group_by(normalized_iupac) %>%
  summarise(
    descriptor_ui_list =
      paste(sort(unique(descriptor_ui_list)), collapse = ";"),
    glytoucan_ac_list =
      paste(sort(unique(glytoucan_ac)), collapse = ";"),
    raw_iupac_variants =
      paste(sort(unique(iupac)), collapse = " | "),
    n_variants = n_distinct(iupac),
    .groups = "drop"
  )

glycan_mesh_normalized %>%
  arrange(desc(n_variants)) %>%
  head(20)

cat("Before normalization:", nrow(final_filtered), "\n")
cat("After normalization :", nrow(glycan_mesh_normalized), "\n")
cat("Merged entries       :", 
    nrow(final_filtered) - nrow(glycan_mesh_normalized), "\n")

final_normalized <- glycan_mesh_normalized %>%
  transmute(
    glytoucan_ac = str_split(glytoucan_ac_list, ";", simplify = TRUE)[,1],
    iupac = normalized_iupac,
    descriptor_ui_list = descriptor_ui_list
  )
any(
  sapply(
    strsplit(glycan_mesh_normalized$descriptor_ui_list, ";"),
    function(x) any(duplicated(x))
  )
)

write.csv(final_normalized, "./data/glycan/glytoucan_iupac_mesh.csv", row.names = FALSE)
