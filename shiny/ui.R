library(shiny)
library(plotly)
library(DT)

fluidPage(
  tags$head(tags$title("Glycan-MeSH Enrichment")),
  
  titlePanel(
    div(
      div("Glycan–MeSH Enrichment",
          style = "font-size: 32px; font-weight: bold;"),
      
      tags$p(
        "This web application enables enrichment analysis based on the Glycan Ontology predicted and constructed using GlycoMeSH-BERT. ",
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
      tags$p("Input one glycan ID in a row（exp: G00031MO）.",
             style="font-size: 14px;"),
      
      textAreaInput(
        "gly_input",
        label = NULL,
        value = "G00031MO\nG03819SO\nG04715SK",
        rows = 10,
        placeholder = "Gxxxxxxx\nGyyyyyyy\n..."
      ),
      
      tags$hr(),
      
      numericInput("top_n", "Top N terms per parent (by GeneRatio)", value = 20, min = 1, max = 200),
      checkboxInput("use_universe_gmt", "Universe = all glycans in GMT (recommended)", TRUE),
      
      actionButton("run_analysis", "RUN", class = "btn-primary")
    ),
    
    mainPanel(
      width = 9,
      
      tabsetPanel(
        tabPanel(
          "Results",
          uiOutput("parent_tabs")
        ),
        tabPanel(
          "All results table",
          DTOutput("all_table"),
          downloadButton("download_all", "Download CSV")
        )
      )
    )
  )
)