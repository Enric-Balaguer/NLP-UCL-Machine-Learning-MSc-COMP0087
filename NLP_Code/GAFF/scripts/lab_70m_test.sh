#!/bin/bash

CONTAINER_IMAGE=gpt-neox.sif
RUN_CONFIG=active_forgetting_configs/lab_70m_test.env

REPOSITORY_LOCATION=$(git rev-parse --show-toplevel)


singularity exec --nv --env "CXX=g++" --env "REPOSITORY_LOCATION=${REPOSITORY_LOCATION}" --env-file "${RUN_CONFIG}" "${REPOSITORY_LOCATION}/${CONTAINER_IMAGE}" /bin/bash -c "cd ${REPOSITORY_LOCATION} && python active_forgetting_run.py"