#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --export=NONE                                    

unset SLURM_EXPORT_ENV                                                  
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

# jobs always start in $HOME: change to a temporary job dir on $WORK
OUTPUTDIR="$WORK/$SLURM_JOB_ID"
mkdir "$OUTPUTDIR"

# datadir WORKDIR="$TMPDIR/$SLURM_JOBID"
# Load archive and unpack data in $TMPDIR
WORKDIR="$TMPDIR/$SLURM_JOB_ID"
mkdir "$WORKDIR"

# Load necessary modules
module load python
source activate /home/woody/iwb3/iwb3001h/software/privat/conda/envs/dp01

var=$1  
mkdir "$WORKDIR/$var/"
mkdir "$OUTPUTDIR/$var"
cd "$WORKDIR/$var"
#Extract the data archive to the tmp dir
tar -xzf "$WORK/archives/${var}.tar.gz"

# Load Repo Directory
# -d defines the dataset
# -e defines the number of epochs to run the training
cd $HOME/hpc_test/
python train.py -s "$WORKDIR/$var" -o "$OUTPUTDIR/$var" -d $var -e 20 

# clean up
cd $HOME
rm -rf "$TMPDIR/$SLURM_JOB_ID/$var"

