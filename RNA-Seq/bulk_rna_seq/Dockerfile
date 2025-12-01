from python:latest

RUN apt-get update && \
	apt-get install -y wget git unzip cmake samtools rsem

RUN pip3 install multiqc

RUN apt install -y openjdk-17-jdk openjdk-17-jre

RUN git clone https://github.com/alexdobin/STAR

RUN git clone https://github.com/celalp/compute_ontario_rna_seq

RUN mkdir -p /rnaseq

WORKDIR /
CMD [ "bash" ]
