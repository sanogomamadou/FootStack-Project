from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime

class PredictionRequest(BaseModel):
    """Schéma pour une requête de prédiction manuelle"""
    home_team: str
    away_team: str
    home_points_avg_5: Optional[float] = 1.0
    home_goals_for_avg_5: Optional[float] = 1.0
    home_goals_against_avg_5: Optional[float] = 1.0
    home_goal_diff_avg_5: Optional[float] = 0.0
    home_recent_form: Optional[float] = 0.0
    away_points_avg_5: Optional[float] = 1.0
    away_goals_for_avg_5: Optional[float] = 1.0
    away_goals_against_avg_5: Optional[float] = 1.0
    away_goal_diff_avg_5: Optional[float] = 0.0
    away_recent_form: Optional[float] = 0.0
    form_difference: Optional[float] = 0.0
    goal_difference: Optional[float] = 0.0
    h2h_win_rate_home: Optional[float] = 0.5
    h2h_win_rate_away: Optional[float] = 0.5
    h2h_matches_played: Optional[int] = 0
    home_days_rest: Optional[int] = 7
    away_days_rest: Optional[int] = 7
    rest_advantage: Optional[int] = 0
    matchday_importance: Optional[float] = 0.5
    is_weekend: Optional[int] = 0

class SimplePredictionRequest(BaseModel):
    """Schéma simplifié pour prédiction automatique"""
    home_team: str
    away_team: str
    match_date: Optional[datetime] = None

class PredictionResponse(BaseModel):
    """Schéma pour une réponse de prédiction"""
    home_team: str
    away_team: str
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    model_accuracy: float = 0.497
    features_used: Optional[Dict] = None

class HealthResponse(BaseModel):
    """Schéma pour health check"""
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: datetime

class UpcomingPredictionsResponse(BaseModel):
    """Schéma pour les prédictions à venir"""
    count: int
    predictions: List[Dict]

class TeamsResponse(BaseModel):
    """Schéma pour la liste des équipes"""
    count: int
    teams: List[str]


# Dans api/schemas.py - Ajouter à la fin

class PlayerClusterBase(BaseModel):
    player_name: str
    team: str
    position_group: str
    goals_per90: float
    assists_per90: float
    goal_contrib_per90: float
    cluster_id: int
    undervalued_score: float

class UndervaluedPlayerResponse(PlayerClusterBase):
    performance_score: float
    minutes_played: int

class TopPlayer(BaseModel):
    player_name: str
    team: str
    position_group: str
    goals_per90: float
    assists_per90: float

class ClusterAnalysisResponse(BaseModel):
    cluster_id: int
    size: int
    avg_goals_per90: float
    avg_assists_per90: float
    avg_goal_contrib_per90: float
    main_positions: Dict[str, int]
    top_teams: Dict[str, int]
    top_players: List[TopPlayer]
    description: str

class PlayerSearchResponse(PlayerClusterBase):
    minutes_played: int
    nationality: str
    age: Optional[int] = None
    similar_players: Optional[List[Dict]] = None
    cluster_ranking: Optional[str] = None

class SimilarPlayer(BaseModel):
    player_name: str
    team: str
    goals_per90: float
    similarity_score: float