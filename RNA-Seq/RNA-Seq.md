# CCM Bulk Short Read RNA-Seq vignette

## Oveview

The goal of this vignette is to introduce some common concepts in bulk RNA-Seq analysis from start to finish. While this
is
not a comprehensive guide to accomplish all that can be accomplished with RNA-Seq it give a relatively thorough tasting
menu of possible analysis methods.

This is document is based on our experience in analysis RNA-Seq data over the years and is based more of a guideline
than
strict requirements. Your research needs might be different, and you might need to experiment with different methods and
tools.

If you have questions please do not hesitate to contact us at ccm@sicckids.ca


## Table of Contents

1. Library preparation
    1. Considerations
    2. Quality Control
    3. Spike Ins
    3. Randomization
2. FastQ files
    1. Format
    2. Quality control
    3. Adapter removal
    4. Checking for contaminations
3. Genomes
    1. Fasta Files
    2. Annotations, gtf, gff, where to get them
4. Alignment
    1. Picking an aligner
    2. Generating an index
    3. Aligning your reads
5. Alignment QC
    1. Tools, how to use them
    2. Some QC baselines
    3. Visualizing alignments, IGV
6. Gene/transcript quantitation
    1. Counts
    2. When counts are not enough (Expectation Maximization)
    3. Pseudoaligmnet and De-Brujin graphs
7. Differential expression
    1. Basic QC
        1. PCA and dimensionality reduction
    2. Gene/Transcript level
        1. What is multiple testing correction?
    3. Selecting meaningful significance thresholds
    4. Additional variables
    5. Controlling for covariates
    6. Interaction terms
        + Basic introduction to linear modelling
        + What does it mean to control for something?
        + What are interaction terms?
        + Nested models
    7. Controlling for known batch effects with ComBat
    8. Controlling for unknown batch effects with SVA
    9. Some standard visualizations
8. Pathway enrichment analysis
    1. Where to get the annotations
    2. Term enrichment
    3. Gene Set enrichment
9. Splicing analysis
10. Allele Specific expression

## Introduction

RNA-Seq is a transformative technique for studying the transcriptome, enabling researchers to quantify gene expression, identify novel transcripts, detect alternative splicing, and explore allele-specific expression. This vignette provides a comprehensive introduction to the RNA-Seq workflow for research scientists who are experts in their field but new to RNA-Seq. We cover each step outlined in the provided table of contents, using well-established methods, and include explanations, references, and code snippets in bash, R, and Python where appropriate. The goal is to equip you with the knowledge to understand and critically evaluate RNA-Seq experiments, though collaboration with bioinformaticians is recommended for complex analyses.

## 1. Library Preparation

Library preparation transforms RNA into a sequencing-ready library of cDNA fragments. This step is critical, as it determines the quality and type of data generated.

### 1.1 Considerations
- **RNA Extraction Protocol**: Choose between poly(A) selection, which enriches for mRNA but requires high RNA integrity (measured by RNA Integrity Number, RIN), and ribosomal depletion, suitable for degraded samples or non-polyadenylated RNAs (e.g., bacterial RNA). Ribosomal depletion is often used for tissue biopsies ([Conesa et al., 2016](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0881-8)).
- **Strand-Specificity**: Strand-specific libraries, such as those using the dUTP method, preserve information about the RNA’s originating strand, aiding in antisense transcript detection and accurate quantification.
- **Fragment Size**: Fragments are typically <500 bp for Illumina sequencing, impacting sequencing efficiency and analysis accuracy.
- **Sequencing Type**: Paired-end (PE) sequencing provides more information than single-end (SE) sequencing, especially for de novo transcript discovery and isoform analysis.
- **Sequencing Depth**: Depth varies by goal: 5–10 million mapped reads suffice for highly expressed genes, while 100 million may be needed for lowly expressed genes or single-cell RNA-Seq.
- **Replicates**: At least three biological replicates are recommended to account for biological variability and ensure statistical power.

### 1.2 Quality Control
- **RNA Integrity**: Assess using RIN via tools like the [Agilent Bioanalyzer](https://www.agilent.com/en/product/automated-electrophoresis/bioanalyzer-systems). A RIN >7 is ideal for poly(A) selection. For clinical samples this might be too stringent. In those cases DV200 (fraction of fragments that are > 200 nt long) is used.
- Prior to fragmentation the results of bioanlyzer should show very distinct ribosomal RNA peaks for the large and small subunit. If these peaks are not **very obviously** distinguishable there might be significant degradation in total RNA samples and a new sample extraction might be required. 
- **Library Size Distribution**: Verify fragment sizes using a Bioanalyzer to ensure consistency with sequencing platform requirements. 

### 1.3 Spike-Ins
Spike-ins are synthetic RNA transcripts of known sequence and quantity added to samples for normalization and calibration. The [ERCC RNA Spike-In Mix](https://www.thermofisher.com/order/catalog/product/4456740) is widely used, minimizing sequence homology with endogenous transcripts to avoid confounding alignments ([Jiang et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3166838/)). Spike-ins help quantify technical variability and ensure accurate expression measurements.

### 1.4 Randomization
Randomize sample processing order to minimize batch effects, where technical variations (e.g., sequencing run) correlate with biological conditions. This is critical in experimental design to ensure unbiased results. 
If there are other variables make sure that you record them (like different kits, different reagents, different days of collection, staff who collected and processed the samples). RNA-Seq is very prone to 
batch effect and known what kinds of batches are there beforehand can allow us to address these issues more effectively if they arise (see below)

## 2. FastQ Files

FastQ files are the standard format for raw sequencing data, containing sequence reads and their quality scores.

### 2.1 Format
Each FastQ entry has four lines:
1. A sequence identifier starting with '@' (e.g., `@SEQ_ID`).
2. The raw nucleotide sequence.
3. A '+' line, optionally repeating the identifier.
4. Quality scores encoded as ASCII characters, where each character represents a Phred score indicating base call confidence ([Wikipedia: FASTQ Format](https://en.wikipedia.org/wiki/FASTQ_format)).

Example FastQ entry:
```
@SEQ_ID
GATTTGGGGTTCAAAGCAGTATCGATCAAATAGTAAATCCATTTGTTCAACTCACAGTTT
+
!''*((((***+))%%%++)(%%%%).1***-+*''))**55CCF>>>>>>CCCCCCC65
```

### 2.2 Quality Control
Use [FastQC](http://www.bioinformatics.babraham.ac.uk/projects/fastqc/) to assess read quality, checking for low-quality bases, adapter contamination, and sequence duplication. FastQC generates reports on per-base quality, GC content, and overrepresented sequences.

Example bash command for FastQC:
```bash
fastqc input.fastq -o output_directory
```

FastQC also has a graphincal interface that can be used if you do not want to use the command line. 

### 2.3 Checking for Contaminations
Contaminations (e.g., from other organisms) can be detected by aligning reads to known contaminant genomes or using [Kraken](https://ccb.jhu.edu/software/kraken/), which classifies reads based on k-mer matching. This ensures data purity before downstream analysis.

### 2.4 Adapter removal

This step is usually not necessary but in some cases it might provide better alignment (see below) there are many tools that can perform this taks and they usually come with a small list of commonly used adapters. It is also possible to 
add your own adapters if you are using novel tools. Some of the tools you might want to consider are 

+ Cutadapt
+ Trimgalore

## 3. Genomes

Reference genomes and annotations are essential for mapping and interpreting RNA-Seq reads.

### 3.1 Fasta Files
Fasta files store genomic sequences in a text-based format, with each sequence preceded by a '>' followed by an identifier ([UCSC FAQ: Data File Formats](https://genome.ucsc.edu/FAQ/FAQformat.html#format4)). These are used as references for alignment.
Not every species has a genome that is available for download. These Fasta files only contain the sequence of the genome. The general fasta file looks something like this:

```
>chr1 #sequence of chromosome one there is one for each chromosome and contig
AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA
GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA
TCGATCGACTAGCATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA
ATGCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
CTAGCTAGCATGCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
```

### 3.2 Annotations (GTF, GFF)
Gene Transfer Format (GTF) and General Feature Format (GFF) files provide gene and transcript annotations, including exon boundaries and gene IDs. Obtain these from databases like [Ensembl](https://www.ensembl.org/) or [UCSC Genome Browser](https://genome.ucsc.edu/).
These files contain very similar information but are formatted slightly differently. A gtf file might look something like this:

```
# GTF file version 2.2
chr1	havana	gene	11873	14409	.	+	.	gene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202";
chr1	havana	transcript	11873	14409	.	+	.	gene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202";
chr1	havana	exon	11873	12227	.	+	.	gene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; exon_number "1"; exon_id "ENSE00002234944";
chr1	havana	exon	12613	12721	.	+	.	gene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; exon_number "2"; exon_id "ENSE00001948552";
chr1	havana	exon	13221	14409	.	+	.	gene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; exon_number "3"; exon_id "ENSE00002156945";
chr1	ensembl_havana	gene	14404	29570	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201";
chr1	ensembl_havana	transcript	14404	29570	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201";
chr1	ensembl_havana	exon	14404	14501	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "1"; exon_id "ENSE00002172712";
chr1	ensembl_havana	exon	15778	15953	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "2"; exon_id "ENSE00002171844";
chr1	ensembl_havana	exon	16606	16703	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "3"; exon_id "ENSE00002172462";
chr1	ensembl_havana	exon	16857	17055	.	-	. gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "4"; exon_id "ENSE00002172085";
chr1	ensembl_havana	exon	17235	17322	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "5"; exon_id "ENSE00002171223";
chr1	ensembl_havana	exon	17605	17703	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "6"; exon_id "ENSE00002172222";
chr1	ensembl_havana	exon	28734	28836	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "7"; exon_id "ENSE00002171554";
chr1	ensembl_havana	exon	29533	29570	.	-	.	gene_id "ENSG00000227232"; transcript_id "ENST00000488147"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; exon_number "8"; exon_id "ENSE00002171908";
```

Depending on the source of the GTF (and GFF) file the annotations and the ids might be slightly different. It is imperative that you use the same genome and the same
genome annotation for your entire experiement. You cannot compare gene/transcript quantitations (see below) from alignments (see below) that use different genomes (genome versions like hg19 vs hg38). 
If you are interested in comparing your results to previously published papers you will need to either download raw reads from a place like [SRA](https://www.ncbi.nlm.nih.gov/sra) and process them 
the same way you processed your results or process your results as described in that paper using the same annotations and genome versions. 

## 4. Alignment

Alignment maps sequencing reads to a reference genome, accounting for splicing in RNA-Seq data.

### 4.1 Picking an Aligner
Splice-aware aligners handle reads spanning exon junctions. [STAR](https://github.com/alexdobin/STAR) and [HISAT2](https://daehwankimlab.github.io/hisat2/) are widely used for their speed and accuracy. STAR is particularly fast and suitable for large datasets ([Dobin et al., 2013](https://academic.oup.com/bioinformatics/article/29/1/15/207991)).

### 4.2 Generating an Index
Index the reference genome to enable efficient alignment. Each aligner has its own indexing method.

Example bash command for STAR indexing:
```bash
STAR --runMode genomeGenerate --genomeDir /path/to/index --genomeFastaFiles genome.fasta --sjdbGTFfile annotation.gtf --runThreadN 8
```

### 4.3 Aligning Your Reads
Align reads to the indexed genome, producing BAM or SAM files.

Example bash command for STAR alignment:
```bash
STAR --genomeDir /path/to/index --readFilesIn read1.fastq read2.fastq --runThreadN 8 --outSAMtype BAM SortedByCoordinate --outFileNamePrefix sample
```

Both STAR and HISAT2 have a lot of options that can be used to optimize your alignments. Feel free to experiment with some of them to get better results. Keep in mind that alingment is a 
computationally intensive process so if you are not sure about what a specific parameter does, you might want to leave that alone as these tools come with very reasonable default values that are 
determined through years of trial and error. 

After your alignment you will either have a SAM file or more likely a BAM file which contains the same information as a SAM but is in binary and is much smaller. 
A typical same file looks something like this:

```
@HD	VN:1.0	SO:unsorted
@SQ	SN:chr1	LN:249250621
# Alignment section
read1	0	chr1	11873	60	50M	*	0	0	ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT	*	NH:i:1	HI:i:1	AS:i:0	nM:i:0
read2	16	chr1	12613	60	50M	*	0	0	ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT	*	NH:i:1	HI:i:1	AS:i:0	nM:i:0
read3	0	chr1	13221	60	50M	*	0	0	ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT	*	NH:i:1	HI:i:1	AS:i:0	nM:i:0
read4	0	chr1	14404	60	50M	*	0	0	ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT	*	NH:i:1	HI:i:1	AS:i:0	nM:i:0
read5	16	chr1	15778	60	50M	*	0	0	ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT	*	NH:i:1	HI:i:1	AS:i:0	nM:i:0
read6 0  chr1 65419 60 50M * 0 0 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT * NH:i:1 HI:i:1 AS:i:0 nM:i:0
read7 16 chr1 69936 60 50M * 0 0 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT * NH:i:1 HI:i:1 AS:i:0 nM:i:0
read8 0  chr1 71495 60 50M * 0 0 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT * NH:i:1 HI:i:1 AS:i:0 nM:i:0
```

A SAM file has many fields, here is a brief description of them:

Header Section:

+ @HD: Header line.
+ N:1.0: SAM format version.
+ SO:unsorted: Sequence sort order is unsorted.
+ @SQ: Sequence dictionary.
+ SN:chr1: Sequence name (chromosome 1, consistent with your GTF).
+ LN:249250621: Sequence length (example length for chr1).

Alignment Section: This section contains the alignment information for each read. I've created 8 alignment lines, showing reads aligned to the exons from your GTF file.

+ read1, read2, etc.: Unique read identifiers.
+ 0 or 16: SAM flag (0 for forward, 16 for reverse).
+ chr1: Chromosome name (from GTF).
+ 11873, 12613, etc.: 1-based leftmost mapping position (from GTF, corresponding to exon start).
+ 60: Mapping quality (example value).
+ 50M: CIGAR string (50 matches). I've made all reads 50M for simplicity.
+ *: Mate chromosome name (not applicable here, so *).
+ 0: Mate alignment position (not applicable, so 0).
+ 0: Inferred template length (not applicable, so 0).
+ ACGT...: Read sequence (example sequence).
+ *: Base quality scores (not shown, so *).
+ NH:i:1: Number of reported alignments for the query sequence.
+ HI:i:1: 1-based sequence number in the alignment.
+ AS:i:0: Alignment score.
+ nM:i:0: Edit distance to reference.

Some aligners (like STAR) have the option to return sorted bam files, you can specify what kind of sorting you would like. If you want you can perform this task after
alignment using tools like [samtools](https://www.htslib.org/). 

## 5. Alignment QC

Quality control ensures alignment reliability.

### 5.1 Tools
Use [Picard](https://broadinstitute.github.io/picard/), [RSeQC](http://rseqc.sourceforge.net/), or [Qualimap](http://qualimap.conesalab.org/) to assess alignment metrics like mapping rates and read distribution.

### 5.2 QC Baselines
Aim for >80% mapped reads for human genomes, with low multi-mapping rates (<10%). Check for even read distribution across genes to avoid biases ([Conesa et al., 2016](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0881-8)).
If you are using total RNA with ribosomal RNA depletion aim for <10% ribosomal RNA contamination, if you are using poly A pulldown aim for <5%. A high quality poly A pulldown RNA-Seq library usually has about >95% alignment where about >80-90% of the aligned reads
mapping coding sequences and utrs with <3% rRNA. 

In addition to the above you might want to pay attention to PCR duplication rates. Depending on the number of PCR cycles that you used (the fewer the better) you should have <50% PCR duplication. 
This number is not the actual amount of PCR duplication. This is calculated by counting the number of reads that have exact same sequence. Since we are randomly fragmenting our libraries this should be an 
unlikely event but for extremely abundant genes this sometimes happens due to chance. 

### 5.3 Visualizing Alignments
[Integrative Genomics Viewer (IGV)](https://software.broadinstitute.org/software/igv/) visualizes alignments, allowing inspection of specific genomic regions. Load BAM files and reference annotations to view read coverage and splice junctions.

## 6. Gene/Transcript Quantitation

Quantitation estimates gene or transcript expression levels.

### 6.1 Counts
Tools like [featureCounts](http://subread.sourceforge.net/) or [HTSeq-count](https://htseq.readthedocs.io/) count reads mapping to genes.

Example bash command for featureCounts:
```bash
featureCounts -a annotation.gtf -o counts.txt aligned.bam
```

### 6.2 When Counts Are Not Enough (Expectation Maximization)
For isoform-level quantification, Expectation Maximization (EM) algorithms like [RSEM](https://deweylab.github.io/RSEM/) or [Cufflinks](http://cole-trapnell-lab.github.io/cufflinks/) estimate transcript abundances by resolving multi-mapping reads ([Trapnell et al., 2012](https://www.nature.com/articles/nprot.2012.016)).

### 6.3 Pseudoalignment and De-Bruijn Graphs
Pseudoalignment methods like [Kallisto](https://pachterlab.github.io/kallisto/) and [Salmon](https://combine-lab.github.io/salmon/) use De-Bruijn graphs for fast, alignment-free quantification, offering high accuracy with lower computational cost ([Bray et al., 2016](https://www.nature.com/articles/nbt.3519)).

Example bash command for Kallisto:
```bash
kallisto quant -i index -o output -b 100 read1.fastq read2.fastq
```

These tools are much faster than STAR or HISAT2 but as the name suggests they do not provide proper SAM files. If your downstream tasks might need processing the raw alignments you might want to consider using proper aligments. In 
terms of quantitation accuracy all the methods mentioned here perform similarly. 

## 7. Differential Expression

Differential expression (DE) analysis identifies genes with significant expression changes between conditions.

### 7.1 Basic QC
- **PCA and Dimensionality Reduction**: Use Principal Component Analysis (PCA) to visualize sample clustering and detect outliers. This ensures samples group by biological condition rather than technical artifacts.

Example R code for PCA with DESeq2:
```r
library(DESeq2)
dds <- DESeqDataSetFromMatrix(countData = countMatrix, colData = sampleTable, design = ~ condition)
vsd <- vst(dds, blind=TRUE)
plotPCA(vsd, intgroup="condition")
```

### 7.2 Gene/Transcript Level
Tools like [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) and [edgeR](https://bioconductor.org/packages/release/bioc/html/edgeR.html) model count data using negative binomial distributions for robust DE analysis ([Love et al., 2014](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8)).

Example R code for DESeq2:
```r
dds <- DESeq(dds)
results <- results(dds, contrast=c("condition","treated","control"))
```

### 7.3 Multiple Testing Correction
Correct for multiple testing using the Benjamini-Hochberg method to control the False Discovery Rate (FDR). An adjusted p-value <0.05 is typically considered significant.

### 7.4 Selecting Meaningful Significance Thresholds
Genes with an adjusted p-value <0.05 and a fold change >2 are often deemed biologically significant, though thresholds depend on the study’s goals.

### 7.5 Additional Variables
Incorporate covariates (e.g., age, sex) into the DE model to account for confounding factors.

### 7.6 Controlling for Covariates
Linear models in DESeq2 or edgeR can include covariates to adjust for known variables.

Example R code with covariates:
```r
design(dds) <- ~ covariate + condition
```

### 7.7 Interaction Terms
Interaction terms test whether the effect of one variable (e.g., treatment) depends on another (e.g., time). These are included in the model design.

### 7.8 Controlling for Known Batch Effects with ComBat
[ComBat](https://bioconductor.org/packages/release/bioc/html/sva.html) corrects for known batch effects by adjusting expression data based on batch identifiers.

Example R code for ComBat:
```r
library(sva)
adjusted_data <- ComBat(dat=exprs, batch=batch)
```

### 7.10 Controlling for Unknown Batch Effects with SVA
[Surrogate Variable Analysis (SVA)](https://bioconductor.org/packages/release/bioc/html/sva.html) identifies and corrects for unknown batch effects by estimating surrogate variables.

Example R code for SVA:
```r
svobj <- sva(dat=exprs, mod=model.matrix(~condition))
```

### 7.11 Standard Visualizations
Common visualizations include:
- **Volcano Plots**: Show log fold change vs. -log10 p-value.
- **MA Plots**: Display log fold change vs. mean expression.
- **Heatmaps**: Visualize expression patterns across samples.

Example R code for a volcano plot:
```r
library(ggplot2)
ggplot(results, aes(x=log2FoldChange, y=-log10(padj))) + geom_point()
```

## 8. Pathway Enrichment Analysis

Pathway enrichment identifies biological pathways overrepresented in differentially expressed genes.

### 8.1 Where to Get Annotations
Use databases like [Gene Ontology (GO)](http://geneontology.org/), [KEGG](https://www.genome.jp/kegg/), or [Reactome](https://reactome.org/).

### 8.2 Term Enrichment
Tools like [DAVID](https://david.ncifcrf.gov/) and [Enrichr](https://maayanlab.cloud/Enrichr/) perform overrepresentation analysis to identify enriched terms.

### 8.3 Gene Set Enrichment
[Gene Set Enrichment Analysis (GSEA)](https://www.gsea-msigdb.org/gsea/index.jsp) analyzes ranked gene lists to detect coordinated pathway changes.

## 9. Splicing Analysis

Alternative splicing analysis detects differences in splice variants between conditions. Tools like [rMATS](http://rnaseq-mats.sourceforge.net/) and [MAJIQ](https://majiq.biociphers.org/) quantify splicing events, such as exon skipping or intron retention ([Shen et al., 2014](https://www.pnas.org/content/111/51/E5593)).

## 10. Allele-Specific Expression

Allele-specific expression (ASE) analysis identifies preferential expression of one allele over another, often requiring phased genotype data. Tools like [WASP](https://github.com/bmvdgeijn/WASP) or [MBASED](https://bioconductor.org/packages/release/bioc/html/MBASED.html) are used, leveraging BAM files and variant calls to quantify allele-specific reads.

