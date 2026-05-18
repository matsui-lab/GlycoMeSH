library(shiny)
library(plotly)
library(DT)

fluidPage(
  tags$head(tags$title("Glycan-MeSH Enrichment Analysis (GlycoMeSH-EA)")),
  
  titlePanel(
    div(
      div(
        "Glycan-MeSH Enrichment Analysis (GlycoMeSH-EA)",
        style = "font-size: 32px; font-weight: bold;"
      ),
      
      tags$p(
        "This web application enables enrichment analysis based on the GlycoMeSH-DB predicted and constructed using GlycoMeSH-BERT. ",
        "Glycans derived from glycomics or glycoproteomics experiments can be used as input. ",
        "For further methodological details, please refer to the corresponding publication below.",
        style = "font-size: 18px; margin-top: 10px;"
      )
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      tags$h4("Input glycans"),
      tags$p(
        "Input one glycan ID per row (e.g., G00031MO).",
        style = "font-size: 14px;"
      ),
      
      textAreaInput(
        "gly_input",
        label = NULL,
        value = "G00031MO\nG03819SO\nG04715SK",
        rows = 10,
        placeholder = "Gxxxxxxx\nGyyyyyyy\n..."
      ),
      
      tags$hr(),
      
      uiOutput("cosine_filter_ui"),
      tags$p(
        "The default lower threshold is 0.3. A threshold around 0.4 showed the highest F1 score in our evaluation. Higher thresholds provide more reliable glycan–MeSH associations but reduce coverage.",
        style = "font-size: 13px; color: #555; margin-top: 5px;"
      ),
      
      numericInput(
        "top_n",
        "Top N terms per parent (by GeneRatio)",
        value = 20,
        min = 1,
        max = 200
      ),
      
      checkboxInput(
        "use_universe_db",
        "Universe = all glycans in filtered database (recommended)",
        TRUE
      ),
      
      actionButton(
        "run_analysis",
        "RUN",
        class = "btn-primary"
      )
    ),
    
    mainPanel(
      width = 9,
      
      tabsetPanel(
        tabPanel(
          "Results",
          uiOutput("parent_tabs"),
        ),
        
        tabPanel(
          "All results table",
          DTOutput("all_table"),
          downloadButton("download_all", "Download CSV")
        ),
        
        tabPanel(
          "Glycan-MeSH sources",
          tags$h4("Evidence & Prediction Scores"),
          tags$p(
            "This table shows source PMID lists for glycan–MeSH pairs associated with the input glycans. ",
            "Each row corresponds to one glycan–MeSH pair, and the sources column contains the PMID list supporting that pair.",
            style = "font-size: 14px;"
          ),
          DTOutput("source_table"),
          downloadButton("download_sources", "Download sources CSV")
        ),
        tabPanel(
          "Download DB files",
          tags$h4("Download source database files"),
          tags$p(
            "Download the original database files used by this application.",
            style = "font-size: 14px;"
          ),
          downloadButton("download_parent_file", "Download mesh_with_parent.csv"),
          tags$br(),
          tags$br(),
          downloadButton("download_source_file", "Download mesh_descriptor_all_with_pmid.csv")
        )
      )
    )
  )
)
