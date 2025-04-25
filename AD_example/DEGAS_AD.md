# Load required packages

``` r
library(Seurat)
library(reticulate)
library(dplyr)
library(magrittr)
library(ggplot2)
library(GGally)
library(stringr)
library(DESeq2)
library(DEGASv2)
library(rstatix)
library(Matrix)
library(ggpubr)
library(plotly)
```

# Load the scRNA-seq data

``` r
sc_raw_data <- read.csv("scDat.csv", header = TRUE, row.names = 1, sep = ",")
scMeta <- read.csv("scLab.csv", header = TRUE, row.names = 1, sep = ",")
createSCobj(sc_raw_data, "scRNA", scMeta, results_save_dir = "DEGAS")
```

# Load the patient data

``` r
bulk_dataset <- read.csv("patDat.csv", header = TRUE, row.names = 1)
patMeta <- read.csv("patLab.csv", header = TRUE, row.names = 1)
phenotype_df <- data.frame(CERAD = as.integer(patMeta$CERAD <= 2), 
                        Braak = as.integer(patMeta$Braak >= 5),
                        CDR = as.integer(patMeta$CDR >= 3))
```

# Run model for different criterions

``` r
sc_dataset <- readRDS("DEGAS/SC_raw_counts/scRNA.rds")
n_st_classes = length(unique(sc_dataset$oupSample.cellType))
# use CERAD for test
label = "CERAD"

DEGAS_preprocessing(bulk_dataset, phenotype_df[, label], sc_dataset, paste0("DEGAS/", label), st_lab_list = sc_dataset@meta.data$oupSample.cellType, model_type = "categorical", diff_expr_files = c("DEGAS/SC_raw_counts/sc_diff_expr.csv"))
```

    ## No Coordinate exist, save meta dataNo Coordinate exist, save meta dataNo Coordinate exist, save meta data

``` r
for (feature in c("Pat_Diff")) {
    folder_path <- paste0("DEGAS", label, "_", feature, "_", Sys.Date())
    degas_sc_results <- run_DEGAS_SCST(paste0("DEGAS/", label, "/", feature, ".RData"), "ClassClass", "Grubman_MSBB", "cross_entropy", "Wasserstein", folder_path, tot_seeds = 10)
    hazard_df <- as.data.frame(degas_sc_results[2])[, c("hazard_df.oupSample.cellType", "hazard_df.hazard", "hazard_df.oupSample.batchCond")]
    colnames(hazard_df) <- c("cellType", "hazard", "batchCond")
    hazard_df <- hazard_df %>% mutate(batchCond = ifelse(batchCond == "AD", "AD", ifelse(batchCond == "ct", "NC", batchCond)))
    write.csv(hazard_df, paste0(folder_path, "/results.csv"))
}
```

    ## Run submodel 0...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_0/configs.json
    ## load models on cuda:0
    ## Run submodel 1...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_1/configs.json
    ## load models on cuda:0
    ## Run submodel 2...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_2/configs.json
    ## load models on cuda:0
    ## Run submodel 3...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_3/configs.json
    ## load models on cuda:0
    ## Run submodel 4...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_4/configs.json
    ## load models on cuda:0
    ## Run submodel 5...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_5/configs.json
    ## load models on cuda:0
    ## Run submodel 6...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_6/configs.json
    ## load models on cuda:0
    ## Run submodel 7...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_7/configs.json
    ## load models on cuda:0
    ## Run submodel 8...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_8/configs.json
    ## load models on cuda:0
    ## Run submodel 9...
    ## Load ClassClass model...
    ## save the configurations into DEGASCERAD_Pat_Diff_2025-04-25/fold_-1_random_seed_9/configs.json
    ## load models on cuda:0
    ## Finish Run and Eval all models
    ## Aggregate all results

# Visualize the Result

``` r
plot_result <- function(hazard_df, sc_dataset) {

  # Boxplot
  ggplot1 <- ggplot(hazard_df, aes(x = batchCond, y = hazard, fill = batchCond)) + 
    geom_boxplot(position = position_dodge(), width = 0.6, outlier.shape = NA) + 
    scale_fill_manual(values = c("AD" = "#D73027", "NC" = "#4575B4")) +
    ylim(0, 1.2) +
    facet_wrap(~cellType, nrow = 2) +
    labs(
      title = "DEGAS Hazard Score by Cell Type",
      x = "Batch Condition",
      y = "Hazard Score",
      fill = "Condition"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 13),
      axis.text = element_text(size = 11),
      strip.text = element_text(face = "bold"),
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 11)
    )

  # UMAP
  umap_data <- Embeddings(sc_dataset, reduction = "umap")
  umap_df <- as.data.frame(umap_data)
  colnames(umap_df) <- c("UMAP_1", "UMAP_2")
  umap_df$cell <- rownames(umap_df)

  hazard_df$cell <- rownames(hazard_df)
  umap_merged <- left_join(
    umap_df, 
    hazard_df[, c("cell", "hazard", "cellType")], 
    by = "cell"
  )

  ggplot2 <- ggplot(umap_merged, aes(x = UMAP_1, y = UMAP_2, color = hazard, shape = cellType)) +
    geom_point(size = 1.4) +
    scale_color_gradient2(
      low = "#000000", mid = "#999999", high = "#D73027", midpoint = 0.5,
      limits = c(0, 1), name = "Hazard"
    ) +
    labs(
      title = "UMAP Colored by DEGAS Hazard",
      x = "UMAP 1",
      y = "UMAP 2",
      shape = "Cell Type"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 13),
      axis.text = element_text(size = 11),
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 11)
    )

  list(boxplot = ggplot1, umap = ggplot2)
}
```

``` r
CREAD_result <- read.csv("DEGASCERAD_Pat_Diff_2025-04-25/results.csv", row.names = 1)
plot_list <- plot_result(CREAD_result, sc_dataset)
plot_list$boxplot
```

![](figure/box.pdf)

``` r
plot_list$umap
```

![](figure/umap.pdf)
