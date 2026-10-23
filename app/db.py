#db.py

from datetime import datetime,timezone
from sqlmodel import Field, Session, SQLModel, create_engine, select, Column,JSON
from typing import Annotated
from fastapi import Depends, Query
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class LogBase(SQLModel):
    patientData: dict = Field(sa_column=Column(JSON))
    probability: float
    label: int

class PredictionLog(LogBase, table=True):
    id: int|None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda : datetime.now(timezone.utc),
    )

class PredictionLogPublic(LogBase):
    id:int


db_path = f"sqlite:///{(Path(__file__).resolve().parent.parent / 'app' / 'logs.db').as_posix()}"
db_url = os.getenv("DATABASE_URL", db_path)
engine = create_engine(db_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

def create_log(log: PredictionLog, session: SessionDep) -> PredictionLog:
    session.add(log)
    session.commit()
    session.refresh(log)
    return log
    
def read_logs(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
)-> PredictionLogPublic:
    predictions = session.exec(select(PredictionLog).offset(offset).limit(limit)).all()
    return predictions