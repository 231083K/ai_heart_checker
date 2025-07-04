import os
import tempfile
import shutil
from typing import List, Annotated

from fastapi import FastAPI, Request, File, UploadFile, Depends, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from ai_model import diagnose_wfdb_record, diagnose_csv_file
import database
import auth
import schemas

database.create_db_and_tables()
app = FastAPI(title="心拍診断AIチェッカー")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db():
    db = database.SessionLocal()
    try: yield db
    finally: db.close()

def get_user(db: Session, username: str):
    return db.query(database.User).filter(database.User.username == username).first()

# --- ▼▼▼ 修正点1 ▼▼▼ ---
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Cookieからトークンを直接読み取り、ユーザーを認証する"""
    token = request.cookies.get("access_token") # "Bearer "なしのトークンを期待
    if not token:
        return None
    try:
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        token_data = schemas.TokenData(username=username)
    except auth.JWTError:
        return None
    
    user = get_user(db, username=token_data.username)
    return user

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# --- ▼▼▼ 修正点2 ▼▼▼ ---
@app.post("/token")
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    """認証後、'Bearer '接頭辞なしでトークンをCookieに保存する"""
    user = get_user(db, form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ユーザー名またはパスワードが違います",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    # "Bearer "を付けずに、トークンそのものをセットする
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

# (以降のコードは変更なし)
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_user(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = get_user(db, username)
    if db_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "このユーザー名は既に使用されています"})
    hashed_password = auth.get_password_hash(password)
    new_user = database.User(username=username, hashed_password=hashed_password)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie(key="access_token")
    return response

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, current_user: schemas.User | None = Depends(get_current_user)):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("index.html", {"request": request, "current_user": current_user})

@app.post("/diagnose", response_class=HTMLResponse)
async def diagnose_ecg(request: Request, files: List[UploadFile] = File(...), db: Session = Depends(get_db), current_user: schemas.User | None = Depends(get_current_user)):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    # ... (この関数の以降の中身は変更なし) ...
    with tempfile.TemporaryDirectory() as temp_dir:
        source_filename = "N/A"; results = {}
        csv_file = next((f for f in files if f.filename.endswith('.csv')), None)
        if csv_file:
            file_path = os.path.join(temp_dir, csv_file.filename)
            with open(file_path, "wb") as buffer: shutil.copyfileobj(csv_file.file, buffer)
            results = diagnose_csv_file(file_path)
            source_filename = csv_file.filename
        else:
            dat_file = next((f for f in files if f.filename.endswith('.dat')), None)
            hea_file = next((f for f in files if f.filename.endswith('.hea')), None)
            if dat_file and hea_file:
                dat_path = os.path.join(temp_dir, dat_file.filename); hea_path = os.path.join(temp_dir, hea_file.filename)
                with open(dat_path, "wb") as buffer: shutil.copyfileobj(dat_file.file, buffer)
                with open(hea_path, "wb") as buffer: shutil.copyfileobj(hea_file.file, buffer)
                record_path_without_ext = os.path.splitext(dat_path)[0]
                results = diagnose_wfdb_record(record_path_without_ext)
                source_filename = dat_file.filename
            else:
                results = {"error": ".csvファイル、または.datと.heaファイルのペアをアップロードしてください。"}
        if "error" not in results:
            new_analysis = database.AnalysisResult(source_filename=source_filename, total_beats=sum(results['counts'].values()), normal_beats=results['counts'].get('N (正常)', 0), s_beats=results['counts'].get('S (上室性)', 0), v_beats=results['counts'].get('V (心室性)', 0), qf_beats=results['counts'].get('Q/F (その他)', 0), abnormal_percentage=results['summary']['abnormal_percentage'], risk_level=results['summary']['level'], summary_text=results['summary']['text'], owner_id=current_user.id)
            db.add(new_analysis); db.commit(); db.refresh(new_analysis)
            print(f"User '{current_user.username}' saved analysis result with ID: {new_analysis.id}")
    return templates.TemplateResponse("result.html", {"request": request, "results": results, "current_user": current_user})

@app.get("/history", response_class=HTMLResponse)
async def read_history(request: Request, db: Session = Depends(get_db), current_user: schemas.User | None = Depends(get_current_user)):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    analysis_history = db.query(database.AnalysisResult).filter(database.AnalysisResult.owner_id == current_user.id).order_by(database.AnalysisResult.created_at.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "history": analysis_history, "current_user": current_user})