FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USER_NAME="sinumaru"
ARG USER_ID="1000"
ARG USER_GID="1000"
ARG REQS_PATH="/XXX/requirements.txt"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get update && apt-get install -y \
    zsh vim wget sudo git libopencv-dev python3 python3-pip python3-tk x11-apps
RUN userdel ubuntu 
RUN groupadd -g ${USER_GID} ${USER_NAME} &&\
    useradd -m -s /bin/zsh -u ${USER_ID} -g ${USER_GID} ${USER_NAME}
RUN echo ${USER_NAME}:${USER_NAME} | chpasswd &&\
    gpasswd -a ${USER_NAME} sudo &&\
    sudo -u ${USER_NAME} touch /home/${USER_NAME}/.zshrc
COPY ${REQS_PATH} requirements.txt
RUN mkdir -p /etc/pip.conf.d && \
    echo "[global]\nbreak-system-packages = true" > /etc/pip.conf
RUN pip install --no-cache-dir -r requirements.txt

USER ${USER_NAME}
WORKDIR /workspace
ENTRYPOINT ["/bin/zsh"]
