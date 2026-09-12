#! /bin/bash
USER=`whoami`
SCRIPT_DIR=$(cd $(dirname $0) || exit; pwd)
GPU_ID=$1
echo ${GPU_ID}
docker run 	-it --rm \
		-u ${USER} \
		--net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/${USER}/.Xauthority:/root/.Xauthority \
		-v ${SCRIPT_DIR}:/workspace \
		--gpus "device=${GPU_ID}" \
		-p 6006:6006 \
		--shm-size 16G \
		Radio-radiance-field_latent-alignment

