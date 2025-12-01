# collect resources
# fasta, this is an older version but it doesnt matter for our purposes. 

wget -c https://ftp.ensembl.org/pub/release-112/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz
gunzip Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz

# gtf
wget -c https://ftp.ensembl.org/pub/release-112/gtf/saccharomyces_cerevisiae/Saccharomyces_cerevisiae.R64-1-1.112.gtf.gz
gunzip Saccharomyces_cerevisiae.R64-1-1.112.gtf.gz

# transcripts fasta
wget -c https://ftp.ensembl.org/pub/release-112/fasta/saccharomyces_cerevisiae/cdna/Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa.gz
gunzip Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa.gz


# find the files and store them in a variable
files=$(ls reads)

mkdir qc

# for each file run fastqc you can also pass all the files in one go if the path is correct
for file in $files
do
    fastqc -t 1 -q --noextract -o qc reads/$file
done


# this is the star executable
STAR="STAR-2.7.11b/bin/Linux_x86_64/STAR"

# generate star index run genomeGenerate mode with 1 thread, save the results in star_index the fasta file is the dna.toplevel and gtf file 
$STAR --runMode genomeGenerate \
      --runThreadN 1 \
      --genomeDir star_index \
      --genomeFastaFiles Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa \
      --sjdbGTFfile Saccharomyces_cerevisiae.R64-1-1.112.gtf

# genereate rsem index we will not be creating a star or bowtie index because we already did that above
# we will first align with star and then use rsem to quantitate this step is optional, the results from STAR is 
# perfectly acceptable

rsem-prepare-reference --gtf Saccharomyces_cerevisiae.R64-1-1.112.gtf -p 1 Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa rsem_index/rsem_index

# star alignment
# again we will need to align each file individually

# some unused options here that might be needed for other experiments
# --readFilesCommand if they are compressed for example zcat for gzip
# for the whole documentation see here: https://physiology.med.cornell.edu/faculty/skrabanek/lab/angsd/lecture_notes/STARmanual.pdf

for file in $files 
do
    $STAR --runMode alignReads \
      --runThreadN 4 \
      --readFilesIn reads/$file \
      --genomeDir star_index \
      --outFileNamePrefix aligned/${file%.fastq} \
      --sjdbGTFfile Saccharomyces_cerevisiae.R64-1-1.112.gtf \
      --outFilterType BySJout \
      --outSAMstrandField intronMotif \
      --alignSoftClipAtReferenceEnds Yes \
      --quantMode TranscriptomeSAM GeneCounts \
      --twopassMode Basic \
      --outSAMtype BAM Unsorted \
      --outSAMattributes All \
      --outSAMunmapped Within \
      --outSAMprimaryFlag AllBestScore 
done

tx_bams=$(ls aligned | grep "Transcriptome")

for bam in $tx_bams
do
    rsem-calculate-expression --bam \
        --num-threads 8 \
        --no-bam-output \
        --forward-prob 0 \
        aligned/$bam rsem_index/rsem_index rsem/${bam%.fastq*}
done 

#run qc metrics using picard tools
genome_bams=$(ls aligned | grep "Aligned.out.bam")

# we are going to do a bunch of things here
# sort the genome bam by coordinate, this will be necessary for visualizations and other qc metric calculations
# mark duplicates this will tell us if we have a lot of pcr duplicates
# rnaseq metrics, general metrics for rna-seq to check overall librarqy quality
# aligmnet metrics, how much of our reads are aligning to different parts of the genome and are they consistent across libraries
# if we had a paired end read we can also run insert size metrics to see how big on average our insert sizes are and if they are
# comparable across libraries and if the distributions are reasonable (not a lot of weird ourtliers in libraries)

# first we need to make a refflat file from the gtf, I could not find a reliable way to make this file from a 3rd party solution
# so I made my own, this works ok, I've used it many times for different projects, you do not need to run this every time just one per genome/gtf combo

Rscript make_refflat.R -g Saccharomyces_cerevisiae.R64-1-1.112.gtf -s -d Saccharomyces_cerevisiae.R64-1-1.112.db -r Saccharomyces_cerevisiae.R64-1-1.112.refflat

for bam in $genome_bams
do
    samtools sort -@ 4 -m 6G -o aligned/${bam%Aligned.out.bam}.sorted.bam aligned/$bam
    java -jar picard.jar MarkDuplicates I=aligned/${bam%Aligned.out.bam}.sorted.bam O=aligned/${bam%Aligned.out.bam}.mdup.bam \
        ASSUME_SORT_ORDER=coordinate M=qc/${bam%Aligned.out.bam}.dup_metrics.txt
    samtools index aligned/${bam%Aligned.out.bam}.mdup.bam
    java -jar picard.jar CollectAlignmentSummaryMetrics I=aligned/${bam%Aligned.out.bam}.mdup.bam \
        R=Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa O=qc/${bam%Aligned.out.bam}.alignment_metrics.txt
    java -jar picard.jar CollectRnaSeqMetrics STRAND_SPECIFICITY=SECOND_READ_TRANSCRIPTION_STRAND I=aligned/${bam%Aligned.out.bam}.mdup.bam \
        O=qc/${bam%Aligned.out.bam}.rnaseq_metrics.txt REF_FLAT=Saccharomyces_cerevisiae.R64-1-1.112.refflat
done

# we will use this to aggregate stuffs
python -m pip install multiqc --user

multiqc -d qc