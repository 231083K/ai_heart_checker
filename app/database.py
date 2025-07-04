import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

# データベース接続URL (docker-compose.ymlで設定したもの)
DATABASE_URL = "postgresql://user:password@db:5432/heart_db"

# SQLAlchemyのエンジンを作成
engine = create_engine(DATABASE_URL)

# DBセッションを作成するためのクラス
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# テーブル定義のベースクラス
Base = declarative_base()


# --- ユーザー情報を保存するテーブルのモデル定義 ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # UserからAnalysisResultへのリレーションシップ
    results = relationship("AnalysisResult", back_populates="owner")


# --- 診断結果を保存するテーブルのモデル定義 ---
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    source_filename = Column(String, index=True)
    
    # --- ここが前回省略されていたカラムです ---
    total_beats = Column(Integer)
    normal_beats = Column(Integer)
    s_beats = Column(Integer)
    v_beats = Column(Integer)
    qf_beats = Column(Integer)
    abnormal_percentage = Column(Float)
    risk_level = Column(String)
    summary_text = Column(String)
    # -----------------------------------------

    # Userテーブルと紐付けるための外部キー
    owner_id = Column(Integer, ForeignKey("users.id"))

    # AnalysisResultからUserへのリレーションシップ
    owner = relationship("User", back_populates="results")


def create_db_and_tables():
    """アプリケーション起動時にテーブルをDB内に作成する"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")