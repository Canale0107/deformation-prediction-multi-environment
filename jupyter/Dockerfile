FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

ARG DOCKER_UID
ARG DOCKER_GID
ARG DOCKER_USER
ARG DOCKER_PASSWORD

LABEL maintainer.nlab=${DOCKER_USER}

USER root

# 不要なソースリストを削除し、必要なパッケージをインストール
RUN set -x && \
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-dev \
        python3-tk \
        fontconfig \
        cabextract \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# フォントのインストールスクリプトをコピーして実行
COPY ./install_fonts.sh /usr/local/bin/install_fonts.sh
RUN chmod +x /usr/local/bin/install_fonts.sh && /usr/local/bin/install_fonts.sh

# ユーザーとグループの作成およびディレクトリ設定
RUN groupadd -g ${DOCKER_GID} ${DOCKER_USER} && \
    useradd -m --uid ${DOCKER_UID} --gid ${DOCKER_GID} --groups sudo ${DOCKER_USER} && \
    echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd && \
    mkdir -p /tf /home/${DOCKER_USER}/.jupyter && \
    chown -R ${DOCKER_USER}:${DOCKER_USER} /tf /home/${DOCKER_USER}/.jupyter

USER ${DOCKER_USER}

ENV PATH $PATH:/home/${DOCKER_USER}/.local/bin
ENV PYTHONPATH=/tf/workspace/deformation-prediction-multi-environment

# 必要なファイルをコピーしてPythonパッケージをインストール
COPY requirements.txt /tf/workspace/requirements.txt
RUN set -x && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tf/workspace/requirements.txt && \
    python3 -m ipykernel install --user --name=devcontainer-kernel --display-name "Python (devcontainer)"

# フォントのインストール確認
RUN fc-list | grep -E "Arial|Times New Roman"

EXPOSE 6006
EXPOSE 8888