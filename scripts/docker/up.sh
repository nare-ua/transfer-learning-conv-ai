#!/bin/bash

set -euxo pipefail

OPTARGS=""
if [[ $(hostname) == "nipa2020-0909" ]]; then
  echo "running on $(hostname)"
  OPTARGS="-e OPENBLAS_CORETYPE=nehalem"
fi

CONTAINER_NAME="convai"
ALIAS="convai"
DATAROOT=/mnt/data
TMPROOT=/mnt/tmp

docker stop $CONTAINER_NAME||true

docker run --shm-size=1g -d --rm --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=nlpdemo_default --network-alias="$ALIAS" \
  --name="$ALIAS" \
  --ipc=host \
  -v $PWD:/workspace \
  -v /mnt/data/transformers_cache:/root/.cache \
  -v ${TMPROOT}:/mnt/tmp \
  -v ${DATAROOT}:/mnt/data \
  $OPTARGS "$CONTAINER_NAME"
