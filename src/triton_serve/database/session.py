from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

from triton_serve.config import get_settings

engine = create_engine(get_settings().database_url)
session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionLocal = scoped_session(session_factory)
Base = declarative_base()
