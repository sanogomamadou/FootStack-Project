# data_ingest/models.py
from sqlalchemy import Column, Integer, String, Date, DateTime, JSON, ForeignKey, UniqueConstraint , Float, Text
from datetime import datetime
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


# Pour les nouveaux modèles de joueurs et statistiques

class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True, index=True)
    fbref_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    position = Column(String)
    team = Column(String)
    age = Column(Integer)
    nationality = Column(String)
    data = Column(JSONB) 

class PlayerStats(Base):
    __tablename__ = "player_stats"
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey('players.id')) #
    season = Column(String)  
    competition = Column(String) 
    minutes_played = Column(Integer) 
    goals = Column(Integer) 
    assists = Column(Integer) 
    data = Column(JSONB) 


                
class PlayerCluster(Base):
    __tablename__ = "player_clusters"
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    cluster_id = Column(Integer)
    position_group = Column(String)
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
