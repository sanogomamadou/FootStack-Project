# data_ingest/models.py
from sqlalchemy import Column, Integer, String, Date, DateTime, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from .db import Base
from sqlalchemy.dialects.postgresql import JSONB

class Competition(Base):
    __tablename__ = "competitions"
    id = Column(Integer, primary_key=True, index=True)  # id from API
    name = Column(String, nullable=False)
    area_name = Column(String)
    code = Column(String)
    data = Column(JSONB)  # raw payload for flexibility

class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    short_name = Column(String)
    tla = Column(String)
    crest_url = Column(String)
    data = Column(JSONB)

class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, nullable=True)
    utc_date = Column(DateTime)
    status = Column(String)
    matchday = Column(Integer, nullable=True)
    home_team_id = Column(Integer, nullable=True)
    away_team_id = Column(Integer, nullable=True)
    score = Column(JSONB)
    raw = Column(JSONB)  # full object from API

    __table_args__ = (
        UniqueConstraint('id', name='uq_match_id'),
    )
