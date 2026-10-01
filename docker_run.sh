#!/bin/bash

USER=$(whoami)
SCRIPT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
GPU_ID=${1:-0}

echo ${GPU_ID}

docker run -it --rm \
    -u ${USER} \
    --net host \
    -e DISPLAY=$DISPLAY \
    -e TERM=xterm-256color \
    -e 'PS1=\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ ' \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/${USER}/.Xauthority:/root/.Xauthority \
    -v ${SCRIPT_DIR}:/workspace \
    --gpus "device=${GPU_ID}" \
    --shm-size 16G \
    --entrypoint /bin/bash \
    radio-radiance-field_latent-alignment \
    --norc -i
