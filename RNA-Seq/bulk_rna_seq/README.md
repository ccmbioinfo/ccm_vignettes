# Bioinformatics: Analysis of RNA-Sequencing Data

This course focuses on basics of data processing and analysis of bulk short read RNA-Sequencing data. Whie we will mention single cell RNA-Seq and long read sequencing these topics will not be covered in depth. The first half of the class will establish he theoretical foundations of RNA-Seq analysis and build the necessary knowledge and glossary. We will introduce different analysis types one can perform using RNA-Seq data and mention common mistakes, method pitfalls and other gotchas. 
In the second half of the lecture, we will perform simple single condition (WT vs KO) differential expression and GO term enrichment analysis from start to finish. All the files that are needed and will be generated are in the google drive folder shared. The files that are shared are small enough to be downloaded full but the only ones that are absolutely required and cannot be downloaded form elsewhere are the reads folder. 
This course assumes that the participants have some familiarity with R and the Linux shell. The data is small enough be run on a personal computer and each step will complete in a few minutes at most. We will go through each command and discuss the reason for including/not including certain parameters in depth in the second half. 
There are a few software packages that need to be installed beforehand to be able to run all the commands. For Linux and mac users all software can be installed using apt-get and brew respectively. For windows users, please install WSL2 (windows subsystem for Linux) and follow the Linux commands form there. The participants are not required to follow along the code if they choose not to. We will be sharing all the material to assist participants in their own analyses. We will briefly discuss how to structure these analyses for an HPC environment and how to import necessary programs beforehand. 
For those who want to install the necessary packages:


# Installation instructions

For Linux, most distributions come with python3 already installed, for some reason you do not have that 
Install python with:

```bash
sudo apt install python3 

# and install pip with 
python3 -m ensurepip --upgrade

# install multiqc
python3 -m pip install multiqc --user

# install other dependencies
sudo apt install samtools, rsem, openjdk-17-jdk openjdk-17-jre
```

For MacOS first install brew:

```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

#then install the other stuff

brew install python
python -m ensurepip --upgrade
python -m pip install multiqc

brew install samtool
brew install picard-tools
```

For rsem follow the instructions on their github profile. Please not that these are not required and any hpc system will either have these already installed as modules or they can be installed upon request by your system administrators. 

## If you want to follow along on your own computer

Clone this repository

```bash
git clone https://github.com/celalp/compute_ontario_rna_seq

cd compute_ontario_rna_seq
```

### Option 1 docker

In this git repository I have a Dockerfile, you can create your own container using the installation instructions on the Docker [website](https://docs.docker.com/get-started/get-docker/). You can use this to run the commands in real time along with the instructions. If you want you can create a container using `docker build -t celalp/compute_ontario_rna_seq:latest` or pull from the dockerhub using `docker pull celalp/compute_ontario_rna_seq:latest`. However, this requires that you have a computer that you have sudo/administrator priviliges. 

### Option 2 singularity or apptainer

If you do not have this option you can use [apptainer](https://apptainer.org/) or [singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html). After intallation you can pull the container from the docker hub. 


### Option 3 conda

You can install conda following the official instructions [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). 

After installation you can create an environment using the following command:

```bash
conda env create -f environment.yaml
```

the `environment.yaml` file is provided for you. This will create an environment called rnaseq. You can activate it using `conda activate rnaseq`. If 
everything works you should be able to use all the programs in the environment. Keep in mind that if you are using windows you will need to do this using WSL (windows subsystem for linux). 

If you are using the conda environment option you do not need to have the "java -jar" in front of picard. And you can call picard directly using `picard`, you do not need to point it towards a specific jar file. 

## Starting the container

### Docker

```bash
docker run -v .:/data --rm -it --entrypoint bash celalp/compute_ontario_rnaseq:latest
```

This will create an interactive bash session. All the necassary dependencies (except for data) should be in the container. 

### Singularity

```bash 
# pull the container from dockerhub and convert to .sif

singularity pull rnaseq.sif docker://celalp/compute_ontario_rnaseq:latest

# start an interactive bash session
singularity shell rnaseq.sif
```

