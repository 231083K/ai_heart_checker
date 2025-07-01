# app/main.py
import os, tempfile, shutil
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
from ai_model import diagnose_wfdb_record, diagnose_csv_file

app = FastAPI(title="心拍診断AIチェッカー")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/diagnose", response_class=HTMLResponse)
async def diagnose_ecg(request: Request, files: List[UploadFile] = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = next((f for f in files if f.filename.endswith('.csv')), None)
        if csv_file:
            file_path = os.path.join(temp_dir, csv_file.filename)
            with open(file_path, "wb") as buffer: shutil.copyfileobj(csv_file.file, buffer)
            results = diagnose_csv_file(file_path)
            return templates.TemplateResponse("result.html", {"request": request, "results": results})

        dat_file = next((f for f in files if f.filename.endswith('.dat')), None)
        hea_file = next((f for f in files if f.filename.endswith('.hea')), None)
        if dat_file and hea_file:
            dat_path = os.path.join(temp_dir, dat_file.filename)
            hea_path = os.path.join(temp_dir, hea_file.filename)
            with open(dat_path, "wb") as buffer: shutil.copyfileobj(dat_file.file, buffer)
            with open(hea_path, "wb") as buffer: shutil.copyfileobj(hea_file.file, buffer)
            record_path_without_ext = os.path.splitext(dat_path)[0]
            results = diagnose_wfdb_record(record_path_without_ext)
            return templates.TemplateResponse("result.html", {"request": request, "results": results})
        
        results = {"error": ".csvファイル、または.datと.heaファイルのペアをアップロードしてください。"}
        return templates.TemplateResponse("result.html", {"request": request, "results": results})