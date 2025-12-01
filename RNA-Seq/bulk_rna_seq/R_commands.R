# start by installing needed packages

install.packages("BiocMangager")

BiocManager::install(c("DESeq2", "ggplot2", "pheatmap", "tximport", "clusterProfiler", 
                       "AnnotationHub", "patchwork", "dplyr"))


# load all the needed packages

library(dplyr)
library(ggplot2)
library(pheatmap)
library(DESeq2)
library(AnnotationHub)
library(clusterProfiler)
library(patchwork)
library(tximport)

# load the data

#first let's get the metadat, samplenames and genotype
samples<-read.csv("samples.txt", header = T, stringsAsFactors = F, sep="\t")
rownames(samples)<-samples$sample

# associate the metadata with the gene expression data
files<-dir("rsem", full.names = T, pattern = "gene")
names(files)<-rownames(samples)

# load gene expression data
rsem <- tximport(files, type = "rsem", txIn = FALSE, txOut = FALSE, countsFromAbundance = "no", abundanceCol = "TPM", 
                 geneIdCol = "gene_id", txIdCol = "transcript_id(s)", countsCol = "expected_count", 
                 lengthCol = "length")

# load annotations

ah <- AnnotationHub()
query(ah, "EnsDb")
ahDb <- query(ah, pattern = c("Saccharomyces Cerevisiae", "EnsDb", 95))
ahEdb<-ahDb[[1]]

annotations<-as.data.frame(genes(ahEdb, columns = c("gene_id", "gene_name", "gene_biotype")))

orgDb <- query(ah, pattern = c("Saccharomyces Cerevisiae", "OrgDb"))
orgDb<-orgDb[[1]]

# separate count data from normalized aboundace data (TPM) we will use the former for differential expression
# and the latter for visualization

counts<-round(rsem$counts)
tpms<-rsem$abundance

# find all the expressed genes
expressed<-rowSums(tpms > 1) >= 1
tpms<-tpms[expressed, ]
counts<-counts[expressed, ]

# create the DESEq object and describe the experimental question
deseq<-DESeqDataSetFromMatrix(counts, colData = samples, design = ~ genotype)
de<-DESeq(deseq)

# some QC visualization about the whole dataset
vsd <- vst(de, blind=FALSE)
pca<-plotPCA(vsd, intgroup=c("genotype"), returnData=T)

#PCA plot to see how the reduced dimensions look like
# do different genotypes cluster together?
ggplot(pca, aes(x=PC1, y=PC2, color=genotype))+geom_point()+theme_minimal()+
  scale_color_brewer(palette = "Set1")

# same question different method, do different genotypes cluster together when we 
# calculate euclidian distance between them and the perform hiearchical clustering
sampleDists <- as.matrix(dist(t(assay(vsd))))
rownames(samples)<-samples$sample
pheatmap(sampleDists, annotation_col = data.frame(genotype=samples$genotype, row.names =samples$sample), cluster_rows = T, cluster_cols = T, show_rownames = F)


# get the differentially expressed genes
res<-as.data.frame(results(de, contrast = c("genotype", "KO", "WT")))
res$gene_id<-rownames(res)
res<-left_join(res, annotations, by="gene_id")
res$sig<-res$padj<0.05


wt_samples<-samples$sample[samples$genotype=="WT"]
ko_samples<-samples$sample[samples$genotype=="KO"]

# calculate mean expression for visualization
wt_tpm<-rowMeans(log2(tpms[, wt_samples]+1))
ko_tpm<-rowMeans(log2(tpms[, ko_samples]+1))
res$WT<-wt_tpm
res$KO<-ko_tpm


scatter<-ggplot(res, aes(x=WT, y=KO, color=sig))+geom_point()+
  theme_minimal()+scale_color_manual(values=c("black", "red"))+geom_abline(slope = 1, intercept = 0, color="blue")+
  ggtitle("WT vs KO")+xlab("log2(WT_TPM +1)")+ylab("log2(KO_TPM +1)")

volcano<-ggplot(res, aes(x=log2FoldChange, y=-log10(padj), color=sig))+geom_point()+
  theme_minimal()+scale_color_manual(values=c("black", "red"))+ ggtitle("WT vs KO")

scatter+volcano+plot_layout(guides = "collect")

# go term enrichment
sig_genes<-na.omit(res$gene_id[res$sig])
universe<-res$gene_id

go<-enrichGO(gene=sig_genes, universe=universe, OrgDb = orgDb, ont="ALL", keyType = "ENSEMBL")
dots<-dotplot(go)
concept<-cnetplot(go, showCategory = 100)
dots+concept+plot_layout(widths = c(2,5))

# save results
write.table(res, "DE_results.txt", col.names = T, row.names = F, sep = "\t", quote = F)
write.table(as.data.frame(go), "GO_results.txt", col.names = T, row.names = F, sep = "\t", quote = F)
