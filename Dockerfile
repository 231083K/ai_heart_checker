FROM python:3.11-slim

# 環境変数設定
ENV PYTHONUNBUFFERED 1

# 作業ディレクトリを作成・設定
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコンテナにコピー
COPY ./app /app

# FastAPIサーバーの起動ポート
EXPOSE 8000