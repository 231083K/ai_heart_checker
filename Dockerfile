FROM python:3.11-slim

# 環境変数設定
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y fonts-ipafont-gothic && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを作成・設定
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコンテナにコピー
COPY ./app /app

# FastAPIサーバーの起動ポート
EXPOSE 8000