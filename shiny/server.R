library(shiny)
library(dplyr)
library(clusterProfiler)
library(plotly)
library(DT)
library(tibble)

# =========================
# Helper utilities
# =========================

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

parse_gly_input <- function(txt) {
  x <- unlist(strsplit(txt, "\n"))
  x <- trimws(x)
  x <- x[nzchar(x)]
  unique(x)
}

gene_ratio_to_num <- function(x) {
  vapply(x, function(s) {
    sp <- strsplit(s, "/", fixed = TRUE)[[1]]
    if (length(sp) != 2) return(NA_real_)
    as.numeric(sp[1]) / as.numeric(sp[2])
  }, numeric(1))
}

build_parent_term_map <- function(parent_df) {
  parent_df %>%
    distinct(mesh_id, descriptor_name, parent_id, parent_name)
}

make_term_id <- function(mesh_id, descriptor_name) {
  safe_name <- gsub("[^A-Za-z0-9]+", "_", descriptor_name)
  safe_name <- gsub("^_+|_+$", "", safe_name)
  paste0("MESH_", mesh_id, "__", safe_name)
}

term_to_mesh_id <- function(term) {
  sub("^MESH_(D[0-9A-Za-z]+).*", "\\1", term)
}

count_sources <- function(source_string) {
  if (is.na(source_string) || !nzchar(source_string)) {
    return(0L)
  }
  
  x <- as.character(source_string)
  x <- gsub("^\\[|\\]$", "", x)
  x <- gsub("'", "", x)
  x <- gsub('"', "", x)
  x <- gsub("\\s+", "", x)
  
  if (!nzchar(x) || x %in% c("NA", "NaN", "None", "NULL")) {
    return(0L)
  }
  
  vals <- unlist(strsplit(x, ",", fixed = TRUE))
  vals <- vals[nzchar(vals)]
  length(unique(vals))
}

has_sources <- function(x) {
  !is.na(x) &
    nzchar(x) &
    !(x %in% c("NA", "NaN", "None", "NULL", "[]"))
}

count_evidence_sources <- function(mesh_id_i, geneID_i, source_df) {
  genes <- unlist(strsplit(geneID_i, "/", fixed = TRUE))
  
  source_df %>%
    filter(
      mesh_id == mesh_id_i,
      gly_id %in% genes
    ) %>%
    summarise(n = sum(n_sources, na.rm = TRUE), .groups = "drop") %>%
    pull(n)
}

count_evidence_pairs <- function(mesh_id_i, geneID_i, source_df) {
  genes <- unlist(strsplit(geneID_i, "/", fixed = TRUE))
  
  source_df %>%
    filter(
      mesh_id == mesh_id_i,
      gly_id %in% genes,
      has_sources(sources)
    ) %>%
    distinct(gly_id, mesh_id) %>%
    nrow()
}

# =========================
# Server
# =========================

function(input, output, session) {
  
  # ---------- Load static resources once ----------
  parent_file <- "mesh_with_parent.csv"
  source_file <- "mesh_descriptor_all_with_pmid.csv"
  
  parent <- read.csv(parent_file, stringsAsFactors = FALSE)
  stopifnot(all(c("mesh_id", "descriptor_name", "parent_id", "parent_name") %in% colnames(parent)))
  parent_map <- build_parent_term_map(parent)
  
  # ---------- Load Glyco-MeSH DB resource ----------
  source_df <- read.csv(
    source_file,
    stringsAsFactors = FALSE,
    check.names = TRUE
  )
  # ---------- Download original DB files ----------
  output$download_parent_file <- downloadHandler(
    filename = function() {
      "mesh_with_parent.csv"
    },
    content = function(file) {
      file.copy(parent_file, file, overwrite = TRUE)
    }
  )
  
  output$download_source_file <- downloadHandler(
    filename = function() {
      "mesh_descriptor_all_with_pmid.csv"
    },
    content = function(file) {
      file.copy(source_file, file, overwrite = TRUE)
    }
  )
  
  # Remove accidental index columns from pandas/R exports.
  index_like_cols <- names(source_df)[
    grepl("^X$|^X\\.\\d+$|^Unnamed", names(source_df), ignore.case = TRUE)
  ]
  if (length(index_like_cols) > 0) {
    source_df <- source_df %>%
      select(-all_of(index_like_cols))
  }
  
  # Minimal handling of column-name variation.
  if ("descriptor_ui" %in% colnames(source_df) && !"mesh_id" %in% colnames(source_df)) {
    source_df <- source_df %>% rename(mesh_id = descriptor_ui)
  }
  if ("glytoucan_ac" %in% colnames(source_df) && !"gly_id" %in% colnames(source_df)) {
    source_df <- source_df %>% rename(gly_id = glytoucan_ac)
  }
  if ("PMID" %in% colnames(source_df) && !"pmids" %in% colnames(source_df)) {
    source_df <- source_df %>% rename(pmids = PMID)
  }
  if ("pmid" %in% colnames(source_df) && !"pmids" %in% colnames(source_df)) {
    source_df <- source_df %>% rename(pmids = pmid)
  }
  if (!"pmids" %in% colnames(source_df)) {
    source_df$pmids <- ""
  }
  if (!"cosine" %in% colnames(source_df)) {
    source_df$cosine <- NA_real_
  }
  if (!"source_type" %in% colnames(source_df)) {
    source_df$source_type <- NA_character_
  }
  
  required_cols <- c("gly_id", "mesh_id", "descriptor_name")
  missing_cols <- setdiff(required_cols, colnames(source_df))
  if (length(missing_cols) > 0) {
    stop(
      paste0(
        "mesh_descriptor_all_with_pmid.csv is missing required columns: ",
        paste(missing_cols, collapse = ", "),
        "\nCurrent columns are: ",
        paste(colnames(source_df), collapse = ", ")
      )
    )
  }
  
  db_base <- source_df %>%
    mutate(
      gly_id = as.character(gly_id),
      mesh_id = as.character(mesh_id),
      descriptor_name = as.character(descriptor_name),
      source_type = as.character(source_type),
      source_type = ifelse(
        is.na(source_type) | !nzchar(source_type),
        ifelse(is.na(suppressWarnings(as.numeric(cosine))), "observed", "predicted"),
        source_type
      ),
      cosine = suppressWarnings(as.numeric(cosine)),
      sources = as.character(pmids),
      sources = ifelse(is.na(sources), "", sources),
      n_sources = vapply(sources, count_sources, integer(1))
    ) %>%
    filter(
      !is.na(gly_id), nzchar(gly_id),
      !is.na(mesh_id), nzchar(mesh_id),
      !is.na(descriptor_name), nzchar(descriptor_name)
    ) %>%
    select(
      gly_id,
      mesh_id,
      descriptor_name,
      source_type,
      cosine,
      sources,
      n_sources
    ) %>%
    distinct()
  
  cosine_values <- db_base$cosine[is.finite(db_base$cosine)]
  cosine_min <- if (length(cosine_values) > 0) min(cosine_values) else NA_real_
  cosine_max <- if (length(cosine_values) > 0) max(cosine_values) else NA_real_
  
  output$cosine_filter_ui <- renderUI({
    
    cos_vals <- source_df$cosine
    cos_vals <- cos_vals[!is.na(cos_vals)]
    
    if (length(cos_vals) == 0) {
      return(tags$p(
        "No cosine similarity values were found in the database.",
        style = "font-size: 13px; color: #777;"
      ))
    }
    
    cos_min <- floor(min(cos_vals, na.rm = TRUE) * 100) / 100
    cos_max <- ceiling(max(cos_vals, na.rm = TRUE) * 100) / 100
    
    tagList(
      sliderInput(
        "cosine_range",
        "Cosine similarity range",
        min = cos_min,
        max = cos_max,
        value = c(max(0.30, cos_min), cos_max),
        step = 0.01,
        ticks = FALSE
      ),
      
      tags$p(
        "The default lower threshold is 0.3. A threshold around 0.4 showed the highest F1 score in our evaluation. Higher thresholds provide more reliable glycan–MeSH associations but reduce coverage.",
        style = "font-size: 13px; color: #555; margin-top: 5px;"
      )
    )
  })
  
  # ---------- Reactive: input glycans ----------
  query_gly <- reactive({
    parse_gly_input(input$gly_input)
  })
  
  # ---------- Filter database at RUN ----------
  filtered_db <- eventReactive(input$run_analysis, {
    db <- db_base
    
    if (is.finite(cosine_min) && is.finite(cosine_max) && !is.null(input$cosine_range)) {
      cr <- input$cosine_range
      db <- db %>%
        filter(
          is.na(cosine) |
            (cosine >= cr[1] & cosine <= cr[2])
        )
    }
    
    db
  })
  
  # ---------- Glycan-MeSH-source table ----------
  source_res <- eventReactive(input$run_analysis, {
    qg <- query_gly()
    db <- filtered_db()
    qg_in <- intersect(qg, unique(db$gly_id))
    
    db %>%
      filter(
        gly_id %in% qg_in,
        has_sources(sources)
      ) %>%
      arrange(gly_id, source_type, mesh_id)
  })
  
  # ---------- Run analysis ----------
  analysis_res <- eventReactive(input$run_analysis, {
    qg <- query_gly()
    validate(need(length(qg) >= 1, "Please input at least 1 glycan ID."))
    
    db <- filtered_db()
    validate(need(nrow(db) > 0, "No glycan-MeSH associations remain after cosine filtering."))
    
    all_glycans_db <- unique(db$gly_id)
    qg_in <- intersect(qg, all_glycans_db)
    validate(need(
      length(qg_in) >= 1,
      "None of the input glycans exist in the filtered Glyco-MeSH database."
    ))
    
    universe_gly <- if (isTRUE(input$use_universe_db)) all_glycans_db else qg_in
    
    term_df <- db %>%
      transmute(
        term = make_term_id(mesh_id, descriptor_name),
        gene = gly_id,
        mesh_id = mesh_id,
        descriptor_name = descriptor_name
      ) %>%
      distinct()
    
    term2gene <- term_df %>%
      select(term, gene) %>%
      distinct()
    
    term2name <- term_df %>%
      select(term, descriptor_name) %>%
      distinct()
    
    source_for_query <- db %>%
      filter(gly_id %in% qg_in)
    
    egmt <- tryCatch(
      enricher(
        gene          = qg_in,
        TERM2GENE     = term2gene,
        TERM2NAME     = term2name,
        universe      = universe_gly,
        pvalueCutoff  = 1.0,
        qvalueCutoff  = 1.0
      ),
      error = function(e) NULL
    )
    
    if (is.null(egmt) || nrow(egmt@result) == 0) {
      return(list(
        all = tibble(),
        by_parent = list(),
        meta = list(
          query = qg_in,
          universe = universe_gly
        )
      ))
    }
    
    all_df <- as.data.frame(egmt@result) %>%
      as_tibble() %>%
      mutate(
        mesh_id = term_to_mesh_id(ID),
        GeneRatioNum = gene_ratio_to_num(GeneRatio),
        neglog10_padj = -log10(p.adjust + 1e-300)
      )
    
    all_df2 <- all_df %>%
      left_join(parent_map, by = c("mesh_id" = "mesh_id")) %>%
      mutate(
        parent_id = ifelse(is.na(parent_id), "UNMAPPED", parent_id),
        parent_name = ifelse(is.na(parent_name), "Unmapped", parent_name)
      )
    
    all_df2 <- all_df2 %>%
      rowwise() %>%
      mutate(
        n_evidence_sources = count_evidence_sources(
          mesh_id,
          geneID,
          source_for_query
        ),
        n_evidence_glycan_mesh_pairs = count_evidence_pairs(
          mesh_id,
          geneID,
          source_for_query
        )
      ) %>%
      ungroup()
    
    by_parent <- split(all_df2, all_df2$parent_id)
    top_n <- as.integer(input$top_n %||% 20)
    
    by_parent_top <- lapply(by_parent, function(df) {
      df %>%
        arrange(desc(GeneRatioNum)) %>%
        slice_head(n = top_n)
    })
    
    list(
      all = all_df2,
      by_parent = by_parent_top,
      meta = list(query = qg_in, universe = universe_gly)
    )
  })
  
  # ---------- UI: dynamic parent tabs ----------
  output$parent_tabs <- renderUI({
    res <- analysis_res()
    validate(need(!is.null(res), "Click RUN to start."))
    
    if (nrow(res$all) == 0) {
      return(tags$div(
        tags$h4("No enrichment results."),
        tags$p("Try more glycans or adjust the cosine similarity range.")
      ))
    }
    
    parents <- res$by_parent
    parent_ids <- names(parents)
    
    panels <- lapply(parent_ids, function(pid) {
      dfp <- parents[[pid]]
      pname <- unique(dfp$parent_name)
      if (length(pname) != 1) pname <- pname[1]
      
      tabPanel(
        title = paste0(pname, " (", pid, ")"),
        tabsetPanel(
          tabPanel("DotPlot", plotlyOutput(outputId = paste0("plt_", pid), height = "650px")),
          tabPanel("Table", DTOutput(outputId = paste0("tbl_", pid)))
        )
      )
    })
    
    do.call(tabsetPanel, c(list(id = "parent_tabset"), panels))
  })
  
  # ---------- Render per-parent plots & tables ----------
  observeEvent(analysis_res(), {
    res <- analysis_res()
    if (is.null(res) || nrow(res$all) == 0) return(NULL)
    
    parents <- res$by_parent
    
    for (pid in names(parents)) {
      local({
        pid_local <- pid
        dfp <- parents[[pid_local]]
        
        dfp <- dfp %>%
          mutate(
            Description = factor(Description, levels = rev(unique(Description)))
          )
        
        output[[paste0("plt_", pid_local)]] <- renderPlotly({
          validate(need(nrow(dfp) > 0, "No rows for this parent."))
          
          dfp2 <- dfp %>%
            mutate(
              Description_chr = as.character(Description),
              GeneRatioNum_plot = pmax(GeneRatioNum, 1e-3),
              DotSize = GeneRatioNum_plot * 0.5
            )
          
          max_x <- suppressWarnings(max(dfp2$neglog10_padj, na.rm = TRUE))
          max_x <- ifelse(is.finite(max_x), max_x, 1)
          
          if (nrow(dfp2) == 1) {
            plot_ly(
              data = dfp2,
              x = ~neglog10_padj,
              y = ~Description_chr,
              type = "scatter",
              mode = "markers",
              marker = list(
                sizemode = "diameter",
                size = 14,
                color = ~neglog10_padj,
                showscale = TRUE
              ),
              text = ~paste0(
                "Parent: ", parent_name,
                "<br>Term: ", Description_chr,
                "<br>ID: ", ID,
                "<br>GeneRatio: ", GeneRatio,
                "<br>Count: ", Count,
                "<br>p.adjust(FDR): ", signif(p.adjust, 3),
                "<br>Evidence sources: ", n_evidence_sources,
                "<br>Evidence glycan-MeSH pairs: ", n_evidence_glycan_mesh_pairs
              ),
              hoverinfo = "text"
            ) %>%
              layout(
                title = paste0("Top 1 term by GeneRatio (Parent: ", unique(dfp2$parent_name)[1], ")"),
                xaxis = list(title = "-log10(p.adjust)", range = c(0, max_x * 1.1)),
                yaxis = list(title = "", type = "category"),
                margin = list(l = 320)
              )
          } else {
            max_gr <- suppressWarnings(max(dfp2$GeneRatioNum_plot, na.rm = TRUE))
            max_gr <- ifelse(is.finite(max_gr), max_gr, NA_real_)
            sizeref_val <- if (!is.na(max_gr) && max_gr > 0) (max_gr / 40)^2 else 1
            
            plot_ly(
              data = dfp2,
              x = ~neglog10_padj,
              y = ~Description_chr,
              type = "scatter",
              mode = "markers",
              marker = list(
                sizemode = "area",
                size = ~DotSize,
                sizemin = 6,
                color = ~neglog10_padj,
                showscale = TRUE,
                sizeref = sizeref_val
              ),
              text = ~paste0(
                "Parent: ", parent_name,
                "<br>Term: ", Description_chr,
                "<br>ID: ", ID,
                "<br>GeneRatio: ", GeneRatio,
                "<br>Count: ", Count,
                "<br>p.adjust(FDR): ", signif(p.adjust, 3),
                "<br>Evidence sources: ", n_evidence_sources,
                "<br>Evidence glycan-MeSH pairs: ", n_evidence_glycan_mesh_pairs
              ),
              hoverinfo = "text"
            ) %>%
              layout(
                title = paste0("Top ", nrow(dfp2), " terms by GeneRatio (Parent: ", unique(dfp2$parent_name)[1], ")"),
                xaxis = list(title = "-log10(p.adjust)"),
                yaxis = list(title = "", type = "category"),
                margin = list(l = 320)
              )
          }
        })
        
        output[[paste0("tbl_", pid_local)]] <- renderDT({
          datatable(
            dfp %>%
              select(
                parent_id,
                parent_name,
                ID,
                Description,
                GeneRatio,
                GeneRatioNum,
                Count,
                pvalue,
                p.adjust,
                qvalue,
                geneID,
                n_evidence_sources,
                n_evidence_glycan_mesh_pairs,
                mesh_id,
                descriptor_name
              ),
            rownames = FALSE,
            options = list(pageLength = 20, scrollX = TRUE)
          )
        })
      })
    }
  })
  
  # ---------- All results table ----------
  output$all_table <- renderDT({
    res <- analysis_res()
    if (is.null(res) || nrow(res$all) == 0) {
      return(datatable(data.frame()))
    }
    
    datatable(
      res$all %>%
        arrange(p.adjust, desc(GeneRatioNum)) %>%
        select(
          parent_id,
          parent_name,
          ID,
          Description,
          GeneRatio,
          GeneRatioNum,
          Count,
          pvalue,
          p.adjust,
          qvalue,
          geneID,
          n_evidence_sources,
          n_evidence_glycan_mesh_pairs,
          mesh_id,
          descriptor_name
        ),
      rownames = FALSE,
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })
  
  output$download_all <- downloadHandler(
    filename = function() {
      paste0("glycan_mesh_enrichment_all_", Sys.Date(), ".csv")
    },
    content = function(file) {
      res <- analysis_res()
      write.csv(res$all, file, row.names = FALSE)
    }
  )
  
  # ---------- Glycan-MeSH-source table ----------
  output$source_table <- renderDT({
    src <- source_res()
    
    if (is.null(src) || nrow(src) == 0) {
      return(datatable(data.frame()))
    }
    
    datatable(
      src %>%
        select(
          gly_id,
          mesh_id,
          descriptor_name,
          source_type,
          cosine,
          n_sources,
          sources
        ),
      rownames = FALSE,
      filter = "top",
      options = list(
        pageLength = 25,
        scrollX = TRUE
      )
    )
  })
  
  output$download_sources <- downloadHandler(
    filename = function() {
      paste0("glycan_mesh_sources_", Sys.Date(), ".csv")
    },
    content = function(file) {
      src <- source_res()
      write.csv(src, file, row.names = FALSE)
    }
  )
}
