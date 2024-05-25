#!/bin/bash

#$ -l h_rt=48:00:0
#$ -l tmpfs=64G
#$ -pe smp 32
#$ -l gpu=1
#$ -l mem=2G
#$ -ac allow=L

CONTAINER_IMAGE=gpt-neox_v3.sif
RUN_CONFIG=active_forgetting_configs/myriad_70m_baseline.env

REPOSITORY_LOCATION=/home/ucabcfj/comp0087-coursework


singularity exec --nv --env "CXX=g++" --env "REPOSITORY_LOCATION=${REPOSITORY_LOCATION}" --env-file "${REPOSITORY_LOCATION}/${RUN_CONFIG}" "${REPOSITORY_LOCATION}/${CONTAINER_IMAGE}" /bin/bash -c "cd ${REPOSITORY_LOCATION} && python active_forgetting_run.py"