import os
import tempfile
import shutil
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List

# AI診断モジュールをインポート
from ai_model import diagnose_ecg_record

app = FastAPI(title="心拍診断AIチェッカー")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/diagnose", response_class=HTMLResponse)
async def diagnose_ecg(request: Request, files: List[UploadFile] = File(...)):
    # 一時的なディレクトリを作成して、アップロードされたファイルを保存
    with tempfile.TemporaryDirectory() as temp_dir:
        dat_file, hea_file = None, None
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if file.filename.endswith('.dat'):
                dat_file = file_path
            elif file.filename.endswith('.hea'):
                hea_file = file_path

        # .dat と .hea の両方がアップロードされたかチェック
        if not (dat_file and hea_file):
            return templates.TemplateResponse("result.html", {
                "request": request,
                "results": {"error": ".dat と .hea の両方のファイルをアップロードしてください。"}
            })

        # .datファイルのパス（拡張子なし）をAI診断関数に渡す
        record_path_without_ext = os.path.splitext(dat_file)[0]
        results = diagnose_ecg_record(record_path_without_ext)

    # 結果をresult.htmlテンプレートに渡してレンダリング
    return templates.TemplateResponse("result.html", {"request": request, "results": results})