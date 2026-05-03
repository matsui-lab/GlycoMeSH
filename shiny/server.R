library(shiny)
library(dplyr)
library(clusterProfiler)
library(plotly)
library(DT)
library(tibble)

# =========================
# Helper utilities
# =========================

parse_gly_input <- function(txt) {
  x <- unlist(strsplit(txt, "\n"))
  x <- trimws(x)
  x <- x[nzchar(x)]
  unique(x)
}

gene_ratio_to_num <- function(x) {
  # x: "k/n"
  vapply(x, function(s) {
    sp <- strsplit(s, "/", fixed = TRUE)[[1]]
    if (length(sp) != 2) return(NA_real_)
    as.numeric(sp[1]) / as.numeric(sp[2])
  }, numeric(1))
}

build_parent_term_map <- function(parent_df) {
  # parent_df columns:
  # mesh_id, descriptor_name, parent_id, parent_name (and maybe tree numbers)
  # term in GMT is expected to look like: "MESH_D000005__Abdomen"
  # We'll extract mesh_id from term and join to parent.
  
  parent_df %>%
    distinct(mesh_id, descriptor_name, parent_id, parent_name)
}

term_to_mesh_id <- function(term) {
  # expects "MESH_D000005__Abdomen" or similar
  # pull D000005
  sub("^MESH_(D[0-9A-Za-z]+).*", "\\1", term)
}

# =========================
# Server
# =========================

function(input, output, session) {
  
  # ---------- Load static resources once ----------
  gmt_file <- "mesh_descriptor_all.gmt"
  parent_file <- "mesh_with_parent.csv"
  
  gset <- read.gmt(gmt_file)  # term, gene
  stopifnot(all(c("term", "gene") %in% colnames(gset)))
  
  parent <- read.csv(parent_file)
  stopifnot(all(c("mesh_id", "descriptor_name", "parent_id", "parent_name") %in% colnames(parent)))
  
  parent_map <- build_parent_term_map(parent)
  
  all_glycans_gmt <- unique(gset$gene)
  
  # ---------- Reactive: input glycans ----------
  query_gly <- reactive({
    parse_gly_input(input$gly_input)
  })
  
  # ---------- Run analysis ----------
  analysis_res <- eventReactive(input$run_analysis, {
    
    qg <- query_gly()
    validate(need(length(qg) >= 1, "Please input at least 1 glycan ID."))
    
    # keep only glycans that exist in GMT
    qg_in <- intersect(qg, all_glycans_gmt)
    validate(need(length(qg_in) >= 1,
                  "None of the input glycans exist in the GMT universe."))
    
    universe_gly <- if (isTRUE(input$use_universe_gmt)) all_glycans_gmt else qg_in
    
    egmt <- tryCatch(
      enricher(
        gene          = qg_in,
        TERM2GENE     = gset,
        universe      = universe_gly,
        pvalueCutoff  = 1.0,
        qvalueCutoff  = 1.0
      ),
      error = function(e) NULL
    )
    
    if (is.null(egmt) || nrow(egmt@result) == 0) {
      return(list(all = tibble(), by_parent = list(), meta = list(
        query = qg_in, universe = universe_gly
      )))
    }
    
    all_df <- as.data.frame(egmt@result) %>%
      as_tibble() %>%
      mutate(
        mesh_id = term_to_mesh_id(ID),
        GeneRatioNum = gene_ratio_to_num(GeneRatio),
        neglog10_FDR = -log10(p.adjust + 1e-300)
      )
    
    # Join to parent mapping (term mesh_id -> possibly multiple parents)
    # If a term has multiple parents, it will duplicate rows (desired).
    all_df2 <- all_df %>%
      left_join(parent_map, by = c("mesh_id" = "mesh_id"))
    
    # If some terms are missing in parent mapping, group them under "Unmapped"
    all_df2 <- all_df2 %>%
      mutate(
        parent_id = ifelse(is.na(parent_id), "UNMAPPED", parent_id),
        parent_name = ifelse(is.na(parent_name), "Unmapped", parent_name)
      )
    
    # Split by parent
    by_parent <- split(all_df2, all_df2$parent_id)
    
    # For each parent: top N by GeneRatioNum
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
        tags$p("Try more glycans or confirm they exist in the GMT.")
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
        
        # ensure ordering in plot (top term at top)
        dfp <- dfp %>%
          mutate(
            Description = factor(Description, levels = rev(unique(Description)))
          )
        
        output[[paste0("plt_", pid_local)]] <- renderPlotly({
          validate(need(nrow(dfp) > 0, "No rows for this parent."))
          
          dfp2 <- dfp %>%
            mutate(
              Description_chr = as.character(Description),
              GeneRatioNum_plot = pmax(GeneRatioNum, 1e-3)
            )
          
          if (nrow(dfp2) == 1) {
            plot_ly(
              data = dfp2,
              x = ~GeneRatioNum,
              y = ~Description_chr,
              type = "scatter",
              mode = "markers",
              marker = list(
                sizemode = "diameter",
                size = 14,          
                color = ~neglog10_FDR,
                showscale = TRUE
              ),
              text = ~paste0(
                "Parent: ", parent_name,
                "<br>Term: ", Description_chr,
                "<br>ID: ", ID,
                "<br>GeneRatio: ", GeneRatio,
                "<br>Count: ", Count,
                "<br>p.adjust(FDR): ", signif(p.adjust, 3)
              ),
              hoverinfo = "text"
            ) %>%
              layout(
                title = paste0("Top 1 term by GeneRatio (Parent: ", unique(dfp2$parent_name)[1], ")"),
                xaxis = list(title = "GeneRatio", range = c(0, max(dfp2$GeneRatioNum_plot) * 1.1)),
                yaxis = list(title = "", type = "category"),
                margin = list(l = 320)
              )
          } else {
            max_gr <- suppressWarnings(max(dfp2$GeneRatioNum_plot, na.rm = TRUE))
            max_gr <- ifelse(is.finite(max_gr), max_gr, NA_real_)
            sizeref_val <- if (!is.na(max_gr) && max_gr > 0) (max_gr / 40)^2 else 1
            
            plot_ly(
              data = dfp2,
              x = ~GeneRatioNum,
              y = ~Description_chr,
              type = "scatter",
              mode = "markers",
              marker = list(
                sizemode = "area",
                size = ~GeneRatioNum_plot,
                sizemin = 6,
                color = ~neglog10_FDR,
                showscale = TRUE,
                sizeref = sizeref_val
              ),
              text = ~paste0(
                "Parent: ", parent_name,
                "<br>Term: ", Description_chr,
                "<br>ID: ", ID,
                "<br>GeneRatio: ", GeneRatio,
                "<br>Count: ", Count,
                "<br>p.adjust(FDR): ", signif(p.adjust, 3)
              ),
              hoverinfo = "text"
            ) %>%
              layout(
                title = paste0("Top ", nrow(dfp2), " terms by GeneRatio (Parent: ", unique(dfp2$parent_name)[1], ")"),
                xaxis = list(title = "GeneRatio"),
                yaxis = list(title = "", type = "category"),
                margin = list(l = 320)
              )
          }
        })
        
        output[[paste0("tbl_", pid_local)]] <- renderDT({
          datatable(
            dfp %>%
              select(parent_id, parent_name, ID, Description, GeneRatio, GeneRatioNum,
                     Count, pvalue, p.adjust, qvalue, geneID, mesh_id, descriptor_name),
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
        select(parent_id, parent_name, ID, Description, GeneRatio, GeneRatioNum,
               Count, pvalue, p.adjust, qvalue, geneID, mesh_id, descriptor_name),
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
}