# Single Cell RNA Sequencing

This workflow provides an introductory guide to analyzing single-cell RNA sequencing (scRNA-seq) data from raw reads through to differential expression using Seurat in R. This guide is meant to help new users get up and running with single-cell analysis and build a solid foundation for more advanced or tailored workflows.

This workflow assumes there there is a two condition experiment that resulted in fastq files.

This workflow Can be split int 6 sections.

1. Read Counts with Cell Ranger

2. Import into Seurat

3. QC and filter reads

3. Normalize and Integrate Data

4. Dimensional Reduction

5. Cell type annotation

6. Differential Expression



## Read Counts with Cell Ranger
You can read more about how to download references and the software from [here](https://www.10xgenomics.com/support/software/cell-ranger/latest/getting-started)

First you need to download cellranger as well as the reference fasta and gtf, then extract the files from the download. The steps in this section can be done on an [interactive node](https://hpc.ccm.sickkids.ca/w/index.php/Slurm_HPC_Quickstart#How_to_initiate_an_interactive_session_on_Slurm.3F) or in a [slurm job.](https://hpc.ccm.sickkids.ca/w/index.php/Slurm_HPC_Quickstart#Submitting_a_batch_job_.28.22sbatch.22.29)


```bash
# For cellranger, note that the link may change in the future so check cellranger's website for latest release
wget -O cellranger-9.0.1.tar.gz "https://cf.10xgenomics.com/releases/cell-exp/cellranger-9.0.1.tar.gz?Expires=1746856836&Key-Pair-Id=APKAI7S6A5RYOXBWRPDA&Signature=He6a2bgetFiPdmzs3iuNn2h~6HQK~22WvoQaFmGvI3Cpcy8ku0~2kFS5KmRmZybs5KcriSF2MqFy4iABWbi6Ct~xZCi057icHdUDOifI4wdoD3zWsGdYZlr-crMyVAL~6Z0zInPkBJyKWR1ViHC-5an7Tc1CJXiTGc40AP-8qHBAmmUH03nZjUMttnzQEkBZQo5re-h7ex4cXAq-p9nghWl9YDvvVlV6lDv-O4f187OSSg5bR4xs6ODxMder0f~Pi9CUPzrS9B9jbro3hNIZkNV2cY9FVKeIaDllb3rOjakVdoppykKPSrWcSCwQ9IR~vqcbWgY3q7R86OgeZlwpQg__"
tar -xzf cellranger-9.0.1.tar.gz

wget "https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz"
tar -xzf refdata-gex-GRCh38-2024-A.tar.gz
```
```bash
/path/to/cellranger-9.0.1/cellranger mkref --genome=GRCh38 \
  --fasta=/path/to/refdata-gex-GRCh38-2024/fasta/genome.fa \
  --genes=/path/to/refdata-gex-GRCh38-2024/genes/genes.gtf.gz
```

And then run `cellranger count` on your conditions seperately. Recommend to do this in a slurm job. Note there are optional parameters `--localmem` and `--localcores` that you can set based on the resources you have available. 

```bash
# Control
/path/to/cellranger-9.0.1/cellranger count --id=Control \
  --transcriptome=/path/to/refdata-GRCh38 \
  --fastqs=/path/to/control_fastqs \
  --sample=Control
```

```bash
# Treatment
/path/to/cellranger-9.0.1/cellranger count --id=Treatment \
  --transcriptome=/path/to/refdata-GRCh38 \
  --fastqs=/path/to/treatment_fastqs \
  --sample=Treatment
```

## Import into Seurat

The rest of this workflow is done in R, and uses the package [`Seurat`](https://satijalab.org/seurat/). These steps can be run on an rstudio session through the HPC, more information on setting that up can be found [here.](https://hpc.ccm.sickkids.ca/w/index.php/HPC_FAQ#Rstudio_on_the_HPC)

Make sure to use the latest version of R and Seurat, which can done through conda or other means. Ensure that `Seurat`, `dplyr` and `ggplot2` are installed in your current environment, if not download with Seurat or use `install.packages("package_name")`.

```r
library(Seurat)
library(dplyr)
library(ggplot2)
```

Then we can load the cells into R, and combine then in preparation for filtering and some basic QC
```r
# Load Control
ctrl_data <- Read10X(data.dir = "/path/to/Control/outs/filtered_feature_bc_matrix")
ctrl <- CreateSeuratObject(counts = ctrl_data, project = "CTRL", min.cells = 3, min.features = 200)
ctrl$condition <- "Control"

# Load Treatment
treat_data <- Read10X(data.dir = "/path/to/Treatment/outs/filtered_feature_bc_matrix")
treat <- CreateSeuratObject(counts = treat_data, project = "TREAT", min.cells = 3, min.features = 200)
treat$condition <- "Treatment"

# Merge the datasets
combined <- merge(ctrl, y = treat, add.cell.ids = c("CTRL", "TREAT"), project = "2ConditionSC")
```

## QC and filter reads
More information on interpreting the violin plot, specific cutoffs to use and other visualizations can be found [here.](https://satijalab.org/seurat/articles/pbmc3k_tutorial#qc-and-selecting-cells-for-further-analysis)
```r
combined[["percent.mt"]] <- PercentageFeatureSet(combined, pattern = "^MT-")

VlnPlot(combined, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

# Filter cells
combined <- subset(combined, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
```

## Normalize and Integrate Data
More information on integration [here](https://satijalab.org/seurat/articles/integration_introduction), if you do not think you need integration you can just skip the splitting step below and just use the `combined` object.

```r
combined <- NormalizeData(combined)
combined <- FindVariableFeatures(combined, selection.method = "vst", nfeatures = 2000)

# Split by condition
split_obj <- SplitObject(combined, split.by = "condition")
split_obj <- lapply(split_obj, SCTransform)

# Integration
features <- SelectIntegrationFeatures(object.list = split_obj, nfeatures = 3000)
split_obj <- PrepSCTIntegration(object.list = split_obj, anchor.features = features)
anchors <- FindIntegrationAnchors(object.list = split_obj, normalization.method = "SCT", anchor.features = features)
integrated <- IntegrateData(anchorset = anchors, normalization.method = "SCT")
```


## Dimensional Reduction
```r
integrated <- RunPCA(integrated)
integrated <- RunUMAP(integrated, dims = 1:30)
integrated <- FindNeighbors(integrated, dims = 1:30)
integrated <- FindClusters(integrated, resolution = 0.5)

DimPlot(integrated, reduction = "umap", group.by = "condition")
DimPlot(integrated, reduction = "umap", label = TRUE)  # Cluster labels
```

## Cell type annotation

You can look at expression of specific genes across all clusters to determine which cell clusters correspond to specific cell types.

```r
# Example manual marker-based annotation
FeaturePlot(integrated, features = c("CD3D", "MS4A1", "LYZ", "PPBP"))

# Assign identities based on previous bioligical knowledge
new.cluster.ids <- c("T cells", "B cells", "Monocytes", "Platelets", ...)
names(new.cluster.ids) <- levels(integrated)
integrated <- RenameIdents(integrated, new.cluster.ids)
DimPlot(integrated, reduction = "umap", label = TRUE)

```

## Differential Expression

This will perform differential expression within each cell cluster across the two conditions and write a resulting object `de_results`.

```r
# Add back condition metadata
DefaultAssay(integrated) <- "RNA"

# Loop over clusters for DE
cluster_ids <- levels(Idents(integrated))
de_results <- list()

for (cl in cluster_ids) {
  subset_cluster <- subset(integrated, idents = cl)
  de <- FindMarkers(subset_cluster, group.by = "condition", ident.1 = "Treatment", ident.2 = "Control")
  de_results[[cl]] <- de
}

# Access DE results for cluster 0
head(de_results[["T cells"]])


saveRDS(integrated, file = "seurat_integrated.rds")

# This can be done for each cell cluster name to save all the different cluster DE results
write.csv(de_results[["T cells"]], file = "DE_Tcells_Treat_vs_Control.csv")
```