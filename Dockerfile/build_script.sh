#!/bin/bash
WORK_DIR=`pwd`
SCRIPT_DIR=$(cd $(dirname $0) || exit; pwd)
USER_NAME=`whoami`
USER_ID=`id -u`
USER_GID=`id -g`
REQS_PATH=${SCRIPT_DIR}"/requirements.txt"
REQS_PATH=`realpath -s --relative-to=${WORK_DIR} ${REQS_PATH}`

docker build \
  --build-arg USER_NAME=${USER_NAME} \
  --build-arg USER_ID=${USER_ID} \
  --build-arg USER_GID=${USER_GID} \
  --build-arg REQS_PATH=${REQS_PATH} \
  -t radio-radiance-field_latent-alignment .