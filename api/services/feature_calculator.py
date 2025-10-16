import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

class FeatureCalculator:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_team_recent_form(self, team_name: str, date: datetime, lookback_matches: int = 5) -> Dict:
        """Calculer la forme récente d'une équipe"""
        try:
            logger.info(f"  Recherche derniers matchs pour {team_name} avant {date}")
            
            # Récupérer les derniers matchs de l'équipe
            query = text("""
            SELECT 
                home_team, away_team, home_score, away_score, result, date,
                CASE 
                    WHEN home_team = :team_name AND result = 'Home' THEN 3
                    WHEN away_team = :team_name AND result = 'Away' THEN 3  
                    WHEN result = 'Draw' THEN 1
                    ELSE 0
                END as points,
                CASE 
                    WHEN home_team = :team_name THEN home_score
                    ELSE away_score
                END as goals_for,
                CASE 
                    WHEN home_team = :team_name THEN away_score
                    ELSE home_score  
                END as goals_against
            FROM matches_cleaned 
            WHERE (home_team = :team_name OR away_team = :team_name)
            AND date < :date
            ORDER BY date DESC
            LIMIT :limit
            """)
            
            team_matches = pd.read_sql_query(
                query, 
                self.db.bind, 
                params={
                    "team_name": team_name,
                    "date": date,
                    "limit": lookback_matches
                }
            )
            
            logger.info(f"  {len(team_matches)} matchs trouvés pour {team_name}")
            
            if len(team_matches) == 0:
                logger.warning(f"  Aucun match trouvé pour {team_name}, utilisation valeurs par défaut")
                return self._get_default_features()
            
            # Afficher les matchs trouvés pour debug
            for _, match in team_matches.iterrows():
                logger.info(f"     {match['date']} | {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']} | Points: {match['points']}")
            
            # Calculer les moyennes
            points_avg = team_matches['points'].mean()
            goals_for_avg = team_matches['goals_for'].mean() 
            goals_against_avg = team_matches['goals_against'].mean()
            goal_diff_avg = goals_for_avg - goals_against_avg
            recent_form = team_matches['points'].sum()
            
            logger.info(f"  Stats {team_name}: Points={points_avg:.2f}, Buts pour={goals_for_avg:.2f}, Buts contre={goals_against_avg:.2f}")
            
            return {
                'points_avg': float(points_avg),
                'goals_for_avg': float(goals_for_avg),
                'goals_against_avg': float(goals_against_avg),
                'goal_diff_avg': float(goal_diff_avg),
                'recent_form': float(recent_form)
            }
            
        except Exception as e:
            logger.error(f"  Erreur calcul forme {team_name}: {e}")
            return self._get_default_features()
    
    def get_head_to_head(self, home_team: str, away_team: str, date: datetime) -> Dict:
        """Calculer les statistiques des confrontations directes"""
        try:
            logger.info(f"  Recherche H2H: {home_team} vs {away_team}")
            
            query = text("""
            SELECT home_team, away_team, result, date
            FROM matches_cleaned
            WHERE ((home_team = :home_team AND away_team = :away_team) 
                   OR (home_team = :away_team AND away_team = :home_team))
            AND date < :date
            ORDER BY date DESC
            LIMIT 10
            """)
            
            h2h_matches = pd.read_sql_query(
                query, 
                self.db.bind,
                params={
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": date
                }
            )
            
            logger.info(f"  {len(h2h_matches)} matchs H2H trouvés")
            
            if len(h2h_matches) == 0:
                logger.warning("  Aucun H2H trouvé, utilisation valeurs par défaut")
                return {'win_rate_home': 0.5, 'win_rate_away': 0.5, 'matches_played': 0}
            
            # Afficher les H2H pour debug
            for _, match in h2h_matches.iterrows():
                logger.info(f"     {match['date']} | {match['home_team']} vs {match['away_team']} | Résultat: {match['result']}")
            
            # Compter les résultats
            home_wins = len(h2h_matches[
                ((h2h_matches['home_team'] == home_team) & (h2h_matches['result'] == 'Home')) |
                ((h2h_matches['away_team'] == home_team) & (h2h_matches['result'] == 'Away'))
            ])
            
            away_wins = len(h2h_matches[
                ((h2h_matches['home_team'] == away_team) & (h2h_matches['result'] == 'Home')) |
                ((h2h_matches['away_team'] == away_team) & (h2h_matches['result'] == 'Away'))
            ])
            
            draws = len(h2h_matches[h2h_matches['result'] == 'Draw'])
            total_matches = len(h2h_matches)
            
            logger.info(f"  H2H: Home wins={home_wins}, Away wins={away_wins}, Draws={draws}")
            
            return {
                'win_rate_home': home_wins / total_matches if total_matches > 0 else 0.5,
                'win_rate_away': away_wins / total_matches if total_matches > 0 else 0.5,
                'matches_played': int(total_matches)
            }
            
        except Exception as e:
            logger.error(f"  Erreur calcul H2H {home_team} vs {away_team}: {e}")
            return {'win_rate_home': 0.5, 'win_rate_away': 0.5, 'matches_played': 0}
    
    def get_contextual_features(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Calculer les features contextuelles"""
        try:
            logger.info(f"  Calcul features contextuelles pour {home_team} vs {away_team}")
            
            # Jours de repos
            home_rest = self._get_days_rest(home_team, match_date)
            away_rest = self._get_days_rest(away_team, match_date)
            
            logger.info(f"  Jours repos: {home_team}={home_rest}j, {away_team}={away_rest}j")
            
            # Importance de la journée (simplifié)
            matchday_importance = 0.5  # À améliorer avec la saison réelle
            
            # Weekend
            is_weekend = 1 if match_date.weekday() >= 5 else 0
            logger.info(f"  Weekend: {is_weekend} (jour {match_date.weekday()})")
            
            return {
                'home_days_rest': home_rest,
                'away_days_rest': away_rest, 
                'rest_advantage': home_rest - away_rest,
                'matchday_importance': matchday_importance,
                'is_weekend': is_weekend
            }
            
        except Exception as e:
            logger.error(f"  Erreur calcul features contextuelles: {e}")
            return {
                'home_days_rest': 7, 'away_days_rest': 7, 'rest_advantage': 0,
                'matchday_importance': 0.5, 'is_weekend': 0
            }
    
    def _get_days_rest(self, team: str, match_date: datetime) -> int:
        """Calculer les jours de repos d'une équipe"""
        try:
            query = text("""
            SELECT MAX(date) as last_match
            FROM matches_cleaned
            WHERE (home_team = :team OR away_team = :team)
            AND date < :match_date
            """)
            
            result = pd.read_sql_query(
                query, 
                self.db.bind,
                params={
                    "team": team,
                    "match_date": match_date
                }
            )
            
            if result.empty or result['last_match'].iloc[0] is None:
                logger.warning(f"  Aucun match précédent trouvé pour {team}, utilisation 7 jours par défaut")
                return 7  # Default
            
            last_match = pd.to_datetime(result['last_match'].iloc[0])
            days_rest = (match_date - last_match).days
            
            logger.info(f"  {team}: dernier match {last_match.date()}, repos={days_rest}j")
            
            return max(1, min(days_rest, 14))  # Entre 1 et 14 jours
            
        except Exception as e:
            logger.error(f"  Erreur calcul jours repos {team}: {e}")
            return 7
    
    def _get_default_features(self) -> Dict:
        """Retourner des features par défaut"""
        return {
            'points_avg': 1.0,
            'goals_for_avg': 1.0, 
            'goals_against_avg': 1.0,
            'goal_diff_avg': 0.0,
            'recent_form': 0.0
        }
    
    def calculate_all_features(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Calculer toutes les features pour une prédiction"""
        logger.info(f"  Début calcul features pour {home_team} vs {away_team} le {match_date}")
        
        # Forme récente
        logger.info(f"  Calcul forme récente...")
        home_form = self.get_team_recent_form(home_team, match_date)
        away_form = self.get_team_recent_form(away_team, match_date)
        
        # Confrontations directes
        logger.info(f"  Calcul H2H...")
        h2h = self.get_head_to_head(home_team, away_team, match_date)
        
        # Features contextuelles
        logger.info(f"  Calcul contexte...")
        context = self.get_contextual_features(home_team, away_team, match_date)
        
        # Features différentielles
        form_difference = home_form['points_avg'] - away_form['points_avg']
        goal_difference = home_form['goal_diff_avg'] - away_form['goal_diff_avg']
        
        #   ASSEMBLAGE FINAL AVEC LES BONS NOMS DE FEATURES
        features = {
            # Home team features
            'home_points_avg_5': home_form['points_avg'],
            'home_goals_for_avg_5': home_form['goals_for_avg'],
            'home_goals_against_avg_5': home_form['goals_against_avg'],
            'home_goal_diff_avg_5': home_form['goal_diff_avg'],
            'home_recent_form': home_form['recent_form'],
            
            # Away team features  
            'away_points_avg_5': away_form['points_avg'],
            'away_goals_for_avg_5': away_form['goals_for_avg'],
            'away_goals_against_avg_5': away_form['goals_against_avg'],
            'away_goal_diff_avg_5': away_form['goal_diff_avg'],
            'away_recent_form': away_form['recent_form'],
            
            # Differential features
            'form_difference': form_difference,
            'goal_difference': goal_difference,
            
            # Head-to-head features
            'h2h_win_rate_home': h2h['win_rate_home'],
            'h2h_win_rate_away': h2h['win_rate_away'],
            'h2h_matches_played': h2h['matches_played'],
            
            # Contextual features
            'home_days_rest': context['home_days_rest'],
            'away_days_rest': context['away_days_rest'],
            'rest_advantage': context['rest_advantage'],
            'matchday_importance': context['matchday_importance'],
            'is_weekend': context['is_weekend']
        }
        
        logger.info(f"  Features calculées pour {home_team} vs {away_team}")
        logger.info(f"  Résumé: Forme Home={home_form['points_avg']:.2f}, Away={away_form['points_avg']:.2f}")
        logger.info(f"  H2H: {h2h['matches_played']} matchs, Home win rate={h2h['win_rate_home']:.2f}")
        
        # Vérification que toutes les features attendues sont présentes
        expected_features = [
            'home_points_avg_5', 'home_goals_for_avg_5', 'home_goals_against_avg_5',
            'home_goal_diff_avg_5', 'home_recent_form', 'away_points_avg_5',
            'away_goals_for_avg_5', 'away_goals_against_avg_5', 'away_goal_diff_avg_5',
            'away_recent_form', 'form_difference', 'goal_difference',
            'h2h_win_rate_home', 'h2h_win_rate_away', 'h2h_matches_played',
            'home_days_rest', 'away_days_rest', 'rest_advantage', 
            'matchday_importance', 'is_weekend'
        ]
        
        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            logger.error(f"  Features manquantes: {missing_features}")
        else:
            logger.info("  Toutes les features attendues sont présentes")
        
        return features