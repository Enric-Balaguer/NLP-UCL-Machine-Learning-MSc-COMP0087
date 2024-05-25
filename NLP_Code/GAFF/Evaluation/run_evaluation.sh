#!/bin/bash -l

# Name of the job
#$ -N evaluate_160m

# Allowed time to run job hours:mins:seconds
#$ -l h_rt=0:20:0

# Amount of wanted GPUs
#$ -l gpu=1

# Request V100 node only
#$ -ac allow=L

# Amount of wanted CPUs
#$ -pe smp 1

# Amount of desired RAM memory
#$ -l mem=100G

#Join the standard error and standard output.
#$ -j y

#Output the results to this file
#$ -o evaluate_160m.txt

# note: Flash Attention only works on Ampere GPU architectures and newer (only L nodes)

REPOSITORY_LOCATION=$HOME/comp0087-coursework
#REPOSITORY_LOCATION=/cs/student/projects1/ml/2023/folkerts/repos/comp0087-coursework

# both specified relative to the root of the comp0087 repository
APPTAINER_IMAGE=gpt-neox_v3.sif
TRAINING_CONFIG=Finetuning/pythia_160m_finetuning.yml

export NEOX_DATA_PATH=/home/ucabcfj/Scratch/dataset_stuff_2/datasets_tokenised/cc100/cc100en_text_document
export NEOX_CHECKPOINT_PATH=$REPOSITORY_LOCATION/checkpoints

# --nv is for nvidia support
apptainer exec --nv --env CXX=g++ "$REPOSITORY_LOCATION"/"$APPTAINER_IMAGE" /bin/bash -c "cd \"$REPOSITORY_LOCATION\" && gpt-neox/deepy.py gpt-neox/train.py \"$TRAINING_CONFIG\""