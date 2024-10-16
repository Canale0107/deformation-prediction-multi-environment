# ベースイメージとしてTensorFlowの公式イメージを使用 (arm64対応)
FROM tensorflow/tensorflow:2.13.0-jupyter

# 作業ディレクトリを設定
WORKDIR /workspace

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# フォントのインストール
COPY ./install_fonts.sh /usr/local/bin/install_fonts.sh
RUN chmod +x /usr/local/bin/install_fonts.sh && /usr/local/bin/install_fonts.sh
RUN apt-get update && apt-get install -y fonts-noto-cjk

ENV PYTHONPATH=/workspace/deformation-prediction-multi-environment

# 必要なファイルをコピーしてPythonパッケージをインストール
COPY requirements.txt /workspace/requirements.txt
RUN set -x && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /workspace/requirements.txt

# ポートを公開
EXPOSE 6006
EXPOSE 8888
