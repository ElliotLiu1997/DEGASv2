############################################################
## Survival objective comparison for reviewer response
## Binary PFS vs Cox log_neg vs Cox rank_loss
############################################################

.libPaths("/geode2/home/u040/jl370/Carbonate/R_libs")
Sys.setenv(LD_LIBRARY_PATH = "/N/soft/rhel8/deeplearning/Python-3.10.10/lib:$LD_LIBRARY_PATH")

library(reticulate)
use_python("/N/soft/rhel8/deeplearning/Python-3.10.10/python", required = TRUE)
py_config()
DEGAS_python <- import("DEGAS_python")

library(dplyr)
library(tidyr)
library(DESeq2)
library(Seurat)
library(DEGASv2)

data_root <- "/N/project/degas_itcr/DEGAS-datasets/MMRF"
results_dir <- "/N/project/ADNDD/Foundation/DEGAS_torch/simulation/survival"
checkpoint_root <- "/N/project/degas_itcr/DEGAS-datasets/MMRF"

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

############################################################
## Load single-cell reference
############################################################

sc_dataset <- readRDS(paste0(data_root, "/samples_integrated_v3_2.rds"))

############################################################
## Load bulk expression and PFS labels
############################################################

pat_full_df <- read.table(paste0(data_root, "/pat.tsv"))
colnames(pat_full_df) <- substring(colnames(pat_full_df), 1, 9)

pat_surv_lab <- read.csv(paste0(data_root, "/mmrf_IA18_OS_PFS.csv"))
rownames(pat_surv_lab) <- pat_surv_lab$Patient_ID

pat_ids <- sort(intersect(colnames(pat_full_df), rownames(pat_surv_lab)))
bulk_dataset <- pat_full_df[, pat_ids]

phenotype <- pat_surv_lab[pat_ids, c(5, 4)]
colnames(phenotype) <- c("time", "status")
phenotype$time <- as.numeric(phenotype$time)
phenotype$status <- as.numeric(phenotype$status)

stopifnot(all(colnames(bulk_dataset) == rownames(phenotype)))

############################################################
## Rebuild filtered_degs using original Rmd logic
############################################################

deg_files <- list.files(
  path = paste0(data_root, "/DEG_analysis"),
  pattern = "^Cluster.*\\.csv$",
  full.names = TRUE
)

if (length(deg_files) == 0) {
  stop("No DEG files found in: ", paste0(data_root, "/DEG_analysis"))
}

all_degs <- do.call(rbind, lapply(deg_files, function(file) {
  read.csv(file, stringsAsFactors = FALSE)
}))

required_deg_cols <- c("Gene", "percG1", "percG2", "log2FC", "Pval")
missing_deg_cols <- setdiff(required_deg_cols, colnames(all_degs))
if (length(missing_deg_cols) > 0) {
  stop("Missing DEG columns: ", paste(missing_deg_cols, collapse = ", "))
}

filtered_degs <- all_degs %>%
  filter(
    ((percG1 > 0.1) | (percG2 > 0.1)) &
      (log2FC > 1) &
      (Pval < 0.05)
  )

filtered_degs <- filtered_degs[!duplicated(filtered_degs$Gene), ]
rownames(filtered_degs) <- filtered_degs$Gene

cat("Number of filtered sc marker genes:", nrow(filtered_degs), "\n")

############################################################
## Select survival genes using binary 3-year PFS
############################################################

# binary_label <- (phenotype$time < 3 * 365) * phenotype$status + 1
horizon <- 3 * 365

known_3y <- (phenotype$status == 1 & phenotype$time < horizon) |
  (phenotype$time >= horizon)

binary_label <- ifelse(
  phenotype$status == 1 & phenotype$time < horizon,
  1,
  0
)

cat("Known 3-year patients:", sum(known_3y), "\n")
cat("Excluded early-censored:", sum(!known_3y), "\n")
print(table(binary_label[known_3y]))


bulk_counts <- bulk_dataset
bulk_counts[bulk_counts < 0] <- 0
bulk_counts <- apply(bulk_counts, c(1, 2), as.integer)

dds <- DESeqDataSetFromMatrix(
  countData = bulk_counts[, known_3y],
  colData = data.frame(
    id = colnames(bulk_dataset)[known_3y],
    label = as.factor(binary_label[known_3y])
  ),
  design = ~label
)

dds <- DESeq(dds)
surv_de <- na.omit(results(dds))
surv_de <- surv_de[order(surv_de$padj), ]

padj.thresh <- 0.001
surv_de_sig <- surv_de[surv_de$padj < padj.thresh, ]

selected_genes_survival <- intersect(rownames(surv_de_sig), filtered_degs$Gene)
selected_genes_survival <- intersect(selected_genes_survival, rownames(bulk_dataset))
selected_genes_survival <- intersect(selected_genes_survival, rownames(sc_dataset[["RNA"]]))

cat("Number of fixed survival genes:", length(selected_genes_survival), "\n")

if (length(selected_genes_survival) < 10) {
  stop("Too few selected genes. Consider relaxing padj.thresh.")
}

write.csv(
  data.frame(gene = selected_genes_survival),
  file.path(results_dir, "SurvPFS_fixed_selected_genes.csv"),
  row.names = FALSE
)


############################################################
## Generate DEGAS input lists using same selected genes
## Compatible with new DEGASv2 interface
############################################################

## Extract single-cell expression matrix from Seurat object
## Prefer RNA counts if available; otherwise use RNA data.
rna_assay <- sc_dataset[["RNA"]]

if ("counts" %in% SeuratObject::Layers(rna_assay)) {
  sc_expr_mat <- SeuratObject::GetAssayData(
    sc_dataset,
    assay = "RNA",
    layer = "counts"
  )
} else {
  sc_expr_mat <- SeuratObject::GetAssayData(
    sc_dataset,
    assay = "RNA",
    layer = "data"
  )
}

sc_expr_mat <- as.matrix(sc_expr_mat)

common_fixed_genes <- Reduce(
  intersect,
  list(
    selected_genes_survival,
    rownames(bulk_dataset),
    rownames(sc_expr_mat)
  )
)

cat("Number of fixed genes used by DEGAS:", length(common_fixed_genes), "\n")

if (length(common_fixed_genes) < 10) {
  stop("Too few common fixed genes after intersecting bulk and sc matrices.")
}

norm_out <- normalize_counts_with_selected_genes(
  bulk_dataset,
  list(sc_expr_mat),
  common_fixed_genes
)

binary_data <- list(
  patDat = norm_out$patDat[known_3y, , drop = FALSE],
  phenotype = binary_label[known_3y],
  scstDat = norm_out$scstDat,
  st_names_list = norm_out$scstName,
  sclab = as.integer(as.factor(sc_dataset@meta.data$seurat_clusters)) - 1
)

cox_data <- list(
  patDat = norm_out$patDat,
  phenotype = phenotype[, c("time", "status")],
  scstDat = norm_out$scstDat,
  st_names_list = norm_out$scstName,
  sclab = as.integer(as.factor(sc_dataset@meta.data$seurat_clusters)) - 1
)

saveRDS(
  binary_data,
  file.path(results_dir, "SurvPFS_binary_fixedgenes.rds")
)

saveRDS(
  cox_data,
  file.path(results_dir, "SurvPFS_cox_fixedgenes.rds")
)

n_st_classes <- length(unique(binary_data$sclab))

############################################################
## Run three survival formulations
############################################################


## Dry run:
tot_seeds_use <- 1
tot_iters_use <- 500

## Final run:
#tot_seeds_use <- 10
#tot_iters_use <- 500

n_st_classes <- length(unique(binary_data$sclab))
checkpoint_root <- "/N/project/ADNDD/Foundation/DEGAS_torch/simulation/survival"
folder_binary <- paste0(
  checkpoint_root,
  "/checkpoints_MMRF_SurvPFS_binary_ClassClass_cross_entropy_Wasserstein_fixedgenes"
)


degas_binary <- run_DEGAS_SCST(
  data_list = binary_data,
  model_type = "ClassClass",
  data_name = "MMRF_SurvPFS_binary_fixedgenes",
  loss_type = "cross_entropy",
  transfer_type = "Wasserstein",
  model_save_dir = folder_binary,
  tot_seeds = tot_seeds_use,
  tot_iters = tot_iters_use,
  lambda1 = 3.0
)

folder_logneg <- paste0(
  checkpoint_root,
  "/checkpoints_MMRF_SurvPFS_ClassCox_log_neg_Wasserstein_fixedgenes"
)

degas_logneg <- run_DEGAS_SCST(
  data_list = cox_data,
  model_type = "ClassCox",
  data_name = "MMRF_SurPFS_ClassCox_log_neg_fixedgenes",
  loss_type = "log_neg",
  transfer_type = "Wasserstein",
  model_save_dir = folder_logneg,
  tot_seeds = tot_seeds_use,
  tot_iters = tot_iters_use,
  lambda1 = 3.0
)

folder_rank <- paste0(
  checkpoint_root,
  "/checkpoints_MMRF_SurvPFS_ClassCox_rank_loss_Wasserstein_fixedgenes"
)

degas_rank <- run_DEGAS_SCST(
  data_list = cox_data,
  model_type = "ClassCox",
  data_name = "MMRF_SurvPFS_ClassCox_rank_loss_fixedgenes",
  loss_type = "rank_loss",
  transfer_type = "Wasserstein",
  model_save_dir = folder_rank,
  tot_seeds = tot_seeds_use,
  tot_iters = tot_iters_use,
  lambda1 = 3.0
)

############################################################
## Compare cell-level and cluster-level scores
############################################################

library(dplyr)
library(tidyr)
library(stringr)

read_seed_hazards <- function(folder_path, formulation_name) {
  seed_dirs <- list.dirs(folder_path, recursive = FALSE, full.names = TRUE)
  seed_dirs <- seed_dirs[grepl("fold_-1_random_seed_[0-9]+$", seed_dirs)]
  
  if (length(seed_dirs) == 0) {
    stop("No seed folders found in: ", folder_path)
  }
  
  out <- lapply(seed_dirs, function(seed_dir) {
    seed_id <- as.integer(sub(".*random_seed_([0-9]+)$", "\\1", seed_dir))
    
    seed_file <- file.path(seed_dir, "summary.csv")
    if (!file.exists(seed_file)) {
      seed_file <- file.path(seed_dir, "summary_mean.csv")
    }
    if (!file.exists(seed_file)) {
      stop("No summary.csv or summary_mean.csv in: ", seed_dir)
    }
    
    df <- read.csv(seed_file)
    
    if (!"hazard" %in% colnames(df)) {
      stop("Cannot find hazard column in: ", seed_file)
    }
    
    data.frame(
      formulation = formulation_name,
      seed_file = seed_file,
      seed_id = seed_id,
      cell_index = seq_len(nrow(df)),
      hazard = as.numeric(df$hazard)
    )
  })
  
  bind_rows(out)
}

seed_hazards <- bind_rows(
  read_seed_hazards(folder_binary, "Binary_PFS_cross_entropy"),
  read_seed_hazards(folder_logneg, "Cox_log_neg"),
  read_seed_hazards(folder_rank, "Cox_rank_loss")
)

cat("Unique cell_index:", length(unique(seed_hazards$cell_index)), "\n")
cat("Metadata rows:", nrow(sc_dataset@meta.data), "\n")
print(table(seed_hazards$formulation, seed_hazards$seed_id))

########

library(dplyr)
library(tidyr)
library(ggplot2)

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

formulation_levels <- c(
  "Binary_PFS_cross_entropy",
  "Cox_log_neg",
  "Cox_rank_loss"
)

formulation_labels <- c(
  Binary_PFS_cross_entropy = "Binary PFS",
  Cox_log_neg = "Cox log-neg",
  Cox_rank_loss = "Cox rank-loss"
)

formulation_colors <- c(
  "Binary PFS" = "#7A7A7A",
  "Cox log-neg" = "#3B82F6",
  "Cox rank-loss" = "#D73027"
)

read_summary_mean <- function(folder_path, formulation_name) {
  df <- read.csv(file.path(folder_path, "summary_mean.csv"))
  
  data.frame(
    formulation = formulation_name,
    cell_index = df$index + 1,
    hazard = as.numeric(df$hazard)
  )
}

read_summary_seed <- function(folder_path, formulation_name) {
  df <- read.csv(file.path(folder_path, "summary.csv"))
  
  data.frame(
    formulation = formulation_name,
    seed_id = as.integer(df$seed),
    cell_index = df$index + 1,
    hazard = as.numeric(df$hazard)
  )
}

mean_hazards <- bind_rows(
  read_summary_mean(folder_binary, "Binary_PFS_cross_entropy"),
  read_summary_mean(folder_logneg, "Cox_log_neg"),
  read_summary_mean(folder_rank, "Cox_rank_loss")
)

seed_hazards <- bind_rows(
  read_summary_seed(folder_binary, "Binary_PFS_cross_entropy"),
  read_summary_seed(folder_logneg, "Cox_log_neg"),
  read_summary_seed(folder_rank, "Cox_rank_loss")
)

cat("Mean hazards unique cells:", length(unique(mean_hazards$cell_index)), "\n")
cat("Seed hazards unique cells:", length(unique(seed_hazards$cell_index)), "\n")
cat("Metadata rows:", nrow(sc_dataset@meta.data), "\n")
print(table(seed_hazards$formulation, seed_hazards$seed_id))

stopifnot(length(unique(mean_hazards$cell_index)) == nrow(sc_dataset@meta.data))
stopifnot(length(unique(seed_hazards$cell_index)) == nrow(sc_dataset@meta.data))

cluster_vec <- as.character(sc_dataset@meta.data$seurat_clusters)

mean_hazards <- mean_hazards %>%
  mutate(
    formulation = factor(formulation, levels = formulation_levels),
    formulation_label = factor(
      formulation_labels[as.character(formulation)],
      levels = formulation_labels[formulation_levels]
    ),
    seurat_clusters = cluster_vec[cell_index]
  )

seed_hazards <- seed_hazards %>%
  mutate(
    formulation = factor(formulation, levels = formulation_levels),
    formulation_label = factor(
      formulation_labels[as.character(formulation)],
      levels = formulation_labels[formulation_levels]
    ),
    seurat_clusters = cluster_vec[cell_index]
  )

stopifnot(!any(is.na(mean_hazards$seurat_clusters)))
stopifnot(!any(is.na(seed_hazards$seurat_clusters)))

############################################################
## Figure 1: cluster-level correlation heatmap
############################################################

cluster_mean <- mean_hazards %>%
  group_by(formulation_label, seurat_clusters) %>%
  summarise(
    mean_hazard = mean(hazard, na.rm = TRUE),
    .groups = "drop"
  )

cluster_wide <- cluster_mean %>%
  pivot_wider(
    names_from = formulation_label,
    values_from = mean_hazard
  )

label_levels <- formulation_labels[formulation_levels]

cor_mat <- cor(
  as.matrix(cluster_wide[, label_levels]),
  method = "spearman",
  use = "pairwise.complete.obs"
)

heat_df <- expand.grid(
  formulation_1 = label_levels,
  formulation_2 = label_levels,
  stringsAsFactors = FALSE
) %>%
  mutate(
    x_id = match(formulation_1, label_levels),
    y_id = match(formulation_2, label_levels),
    correlation = mapply(
      function(x, y) cor_mat[x, y],
      formulation_1,
      formulation_2
    )
  ) %>%
  filter(y_id >= x_id) %>%
  mutate(
    formulation_1 = factor(formulation_1, levels = label_levels),
    formulation_2 = factor(formulation_2, levels = rev(label_levels)),
    label = sprintf("%.2f", correlation)
  )

p_heat <- ggplot(
  heat_df,
  aes(x = formulation_1, y = formulation_2, fill = correlation)
) +
  geom_tile(color = "white", linewidth = 1.0) +
  geom_text(aes(label = label), size = 6, fontface = "bold") +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Spearman\ncorrelation"
  ) +
  coord_fixed() +
  theme_minimal(base_size = 18) +
  labs(x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(size = 16, angle = 35, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 16, color = "black"),
    panel.grid = element_blank(),
    legend.title = element_text(size = 15),
    legend.text = element_text(size = 13),
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_heat)

ggsave(
  file.path(results_dir, "survival_loss_cluster_correlation_heatmap.pdf"),
  p_heat,
  width = 6,
  height = 5,
  device = cairo_pdf
)

ggsave(
  file.path(results_dir, "survival_loss_cluster_correlation_heatmap.png"),
  p_heat,
  width = 6,
  height = 5,
  dpi = 600
)

############################################################
## Figure 2: seed-to-seed stability
############################################################

cluster_seed_mean <- seed_hazards %>%
  group_by(formulation_label, seed_id, seurat_clusters) %>%
  summarise(
    mean_hazard = mean(hazard, na.rm = TRUE),
    .groups = "drop"
  )

calc_seed_stability <- function(df) {
  seed_wide <- df %>%
    select(seurat_clusters, seed_id, mean_hazard) %>%
    pivot_wider(
      names_from = seed_id,
      values_from = mean_hazard,
      names_prefix = "seed_"
    )
  
  seed_cols <- setdiff(colnames(seed_wide), "seurat_clusters")
  seed_mat <- as.matrix(seed_wide[, seed_cols])
  storage.mode(seed_mat) <- "numeric"
  
  seed_cor <- cor(
    seed_mat,
    method = "spearman",
    use = "pairwise.complete.obs"
  )
  
  idx <- which(upper.tri(seed_cor), arr.ind = TRUE)
  
  data.frame(
    seed_1 = colnames(seed_cor)[idx[, 1]],
    seed_2 = colnames(seed_cor)[idx[, 2]],
    stability = seed_cor[idx]
  )
}

seed_stability <- cluster_seed_mean %>%
  group_by(formulation_label) %>%
  group_modify(~ calc_seed_stability(.x)) %>%
  ungroup() %>%
  mutate(
    formulation_label = factor(
      formulation_label,
      levels = label_levels
    )
  )

p_stability <- ggplot(
  seed_stability,
  aes(x = formulation_label, y = stability, fill = formulation_label)
) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 0.8,
    alpha = 0.9
  ) +
  scale_fill_manual(
    values = formulation_colors,
    drop = FALSE
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw(base_size = 18) +
  labs(
    title = "Seed stability",
    x = NULL,
    y = "Spearman correlation"
  ) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.text.x = element_text(size = 16, angle = 35, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 16, color = "black"),
    axis.title.y = element_text(size = 18, face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.35),
    panel.border = element_rect(color = "black", linewidth = 0.8),
    plot.margin = margin(10, 10, 10, 10)
  )

ggsave(
  file.path(results_dir, "survival_loss_seed_stability_boxplot.pdf"),
  p_stability,
  width = 6,
  height = 5,
  device = cairo_pdf
)

ggsave(
  file.path(results_dir, "survival_loss_seed_stability_boxplot.png"),
  p_stability,
  width = 6,
  height = 5,
  dpi = 600
)

library(Seurat)
library(ggplot2)
library(dplyr)

umap_df <- as.data.frame(Embeddings(sc_dataset, reduction = "umap"))
colnames(umap_df)[1:2] <- c("UMAP_1", "UMAP_2")
umap_df$index <- seq_len(nrow(umap_df)) - 1  # DEGAS index is 0-based

library(Seurat)
library(ggplot2)
library(dplyr)

umap_df <- as.data.frame(Embeddings(sc_dataset, reduction = "umap"))
colnames(umap_df)[1:2] <- c("UMAP_1", "UMAP_2")
umap_df$index <- seq_len(nrow(umap_df)) - 1

plot_one_umap <- function(degas_result, title) {
  plot_df <- umap_df %>%
    left_join(
      degas_result %>%
        select(index, hazard) %>%
        mutate(`Progression Association` = hazard * 2 - 1),
      by = "index"
    )
  
  ggplot(plot_df, aes(UMAP_1, UMAP_2)) +
    geom_point(aes(color = `Progression Association`), size = 0.25, alpha = 0.8) +
    scale_color_gradient2(
      low = "#8DB4D2",
      mid = "#D9D9D9",
      high = "#D73027",
      midpoint = 0,
      limits = c(-1, 1),
      name = "Progression\nAssociation"
    ) +
    coord_equal() +
    labs(
      title = title,
      x = "UMAP 1",
      y = "UMAP 2"
    ) +
    theme_classic(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, hjust = 0.5),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 12)
    )
}

p_binary <- plot_one_umap(degas_binary, "Binary PFS")
p_logneg <- plot_one_umap(degas_logneg, "Cox log-neg")
p_rank   <- plot_one_umap(degas_rank, "Cox rank-loss")

ggsave(file.path(results_dir, "survival_umap_binary_pfs.png"), p_binary, width = 5.5, height = 4.5, dpi = 300)
ggsave(file.path(results_dir, "survival_umap_cox_logneg.png"), p_logneg, width = 5.5, height = 4.5, dpi = 300)
ggsave(file.path(results_dir, "survival_umap_cox_rankloss.png"), p_rank, width = 5.5, height = 4.5, dpi = 300)



