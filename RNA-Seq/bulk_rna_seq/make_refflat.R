library(optparse)

option_list <- list( 
  make_option(c("-g", "--gtf"), action="store", help="gtf file to use"),
  make_option(c("-r", "--refflat"), action="store", help="refflat output"),
  make_option(c("-s", "--savedb"), action="store_true", help="save the intermediate txdb"),
  make_option(c("-d", "--dbname"), action="store", help="name for the txdb")
)

opt <- parse_args(OptionParser(option_list=option_list))

library(GenomicFeatures)
library(dplyr)

db<-makeTxDbFromGFF(opt$gtf)

txdf<-as.data.frame(transcripts(db, columns=c("tx_name", "gene_id")))
txdf$gene_id<-unlist(txdf$gene_id)

#this is the first 5 columns
txdf<-txdf[, c(7,6,1,5,2,3)]

#getcds
cdss<-unlist(cdsBy(db, by="tx", use.names=T))
names(cdss)[is.na(names(cdss))]<-"NA"
cdss$tx_name<-names(cdss)
names(cdss)<-c(1:length(cdss))
cdss<-as.data.frame(cdss)

cdss2<-cdss %>%
  group_by(tx_name) %>%
  mutate(cds_start=min(start)) %>%
  mutate(cds_end=max(end)) %>%
  select(., tx_name, cds_start, cds_end) %>%
  unique(.) %>%
  as.data.frame(.)

exdf<-unlist(exonsBy(db, by="tx", use.name=T))
names(exdf)[is.na(names(exdf))]<-"NA"
exdf$tx_name<-names(exdf)
names(exdf)<-c(1:length(exdf))
exdf<-as.data.frame(exdf)

exdf2 <- exdf %>%
  group_by(tx_name) %>%
  mutate(exon_count=max(exon_rank)) %>%
  mutate(starts=paste0(paste(start, collapse=","), ",")) %>%
  mutate(ends=paste0(paste(end, collapse=","), ",")) %>%
  select(., tx_name, exon_count, starts, ends) %>%
  unique(.) %>%
  as.data.frame(.)

refflat<-left_join(txdf, cdss2, by="tx_name")
refflat<-left_join(refflat, exdf2, by="tx_name")
refflat$cds_start[is.na(refflat$cds_start)]<-refflat$end[is.na(refflat$cds_start)]
refflat$cds_end[is.na(refflat$cds_end)]<-refflat$end[is.na(refflat$cds_end)]
write.table(refflat, opt$refflat, col.names = F, row.names = F, quote = F, sep = "\t")

if(opt$savedb){
  saveDb(db, opt$dbname)
}