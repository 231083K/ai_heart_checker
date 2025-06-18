from fastapi import FastAPI

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# ルートURL ("/") へのGETリクエストを処理するエンドポイント
@app.get("/")
def read_root():
    return {"message": "Hello, AI Heart Checker!"}