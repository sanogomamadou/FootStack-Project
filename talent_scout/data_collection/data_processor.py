# talent_scout/data_collection/data_processor.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processeur de données pour nettoyer et enrichir les données joueurs"""
    
    def __init__(self, raw_data_path: str = "data/players_stats_complete.csv"):
        self.raw_data_path = raw_data_path
        self.df = None
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Charger et nettoyer les données brutes"""
        logger.info("Chargement et nettoyage des données...")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Fichier {self.raw_data_path} non trouvé. Exécutez d'abord le scraper.")
        
        # Charger les données
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Données chargées: {len(self.df)} joueurs")
        
        # Nettoyage des données
        self._clean_data()
        
        # Enrichissement des données
        self._enrich_data()
        
        logger.info("Données nettoyées et enrichies avec succès")
        return self.df
    
    def _clean_data(self):
        """Nettoyer les données brutes"""
        # Supprimer les doublons
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['player_name', 'team', 'season'], keep='first')
        logger.info(f"Doublons supprimés: {initial_count - len(self.df)}")
        
        # Nettoyer les positions
        self.df['position'] = self.df['position'].fillna('Unknown')
        
        # Nettoyer l'âge
        self.df['age'] = self.df['age'].fillna(self.df['age'].median())
        self.df['age'] = self.df['age'].astype(int)
        
        # Nettoyer les minutes jouées
        self.df['minutes_played'] = self.df['minutes_played'].fillna(0)
        
        # Filtrer les joueurs avec au moins 90 minutes
        self.df = self.df[self.df['minutes_played'] >= 90]
        logger.info(f"Joueurs avec +90 minutes: {len(self.df)}")
    
    def _enrich_data(self):
        """Enrichir les données avec des métriques avancées"""
        logger.info("Enrichissement des données avec métriques avancées...")
        
        # Métriques de performance
        self.df['total_contributions'] = self.df['goals'] + self.df['assists']
        self.df['contributions_per90'] = (self.df['total_contributions'] / self.df['minutes_played']) * 90
        
        # Efficacité de tir (estimation)
        self.df['shot_efficiency'] = self.df['goals'] / (self.df['goals'] + 10)  # Approximation
        
        # Catégorisation par position
        self.df['position_group'] = self.df['position'].apply(self._categorize_position)
        
        # Score de performance global
        self.df['performance_score'] = self._calculate_performance_score()
        
        # Catégorie d'âge
        self.df['age_category'] = self.df['age'].apply(self._categorize_age)
        
        # Niveau de ligue (pondération)
        self.df['league_weight'] = self.df['league'].apply(self._get_league_weight)
        
        logger.info("Données enrichies avec succès")
    
    def _categorize_position(self, position: str) -> str:
        """Catégoriser les positions en groupes"""
        position = str(position).upper()
        
        if any(pos in position for pos in ['GK', 'GOALKEEPER']):
            return 'Gardien'
        elif any(pos in position for pos in ['DF', 'DEFENDER', 'CB', 'FB', 'RB', 'LB']):
            return 'Défenseur'
        elif any(pos in position for pos in ['MF', 'MIDFIELDER', 'CM', 'DM', 'AM']):
            return 'Milieu'
        elif any(pos in position for pos in ['FW', 'FORWARD', 'ST', 'CF', 'WF']):
            return 'Attaquant'
        else:
            return 'Autre'
    
    def _categorize_age(self, age: int) -> str:
        """Catégoriser les joueurs par âge"""
        if age <= 21:
            return 'Jeune (≤21)'
        elif age <= 25:
            return 'Jeune Adulte (22-25)'
        elif age <= 29:
            return 'Prime (26-29)'
        else:
            return 'Vétéran (30+)'
    
    def _get_league_weight(self, league: str) -> float:
        """Pondération des ligues pour le scoring"""
        weights = {
            'Premier League': 1.0,
            'La Liga': 0.95,
            'Bundesliga': 0.9,
            'Serie A': 0.9,
            'Ligue 1': 0.85
        }
        return weights.get(league, 0.8)
    
    def _calculate_performance_score(self) -> float:
        """Calculer un score de performance global"""
        # Normaliser les métriques
        goals_norm = (self.df['goals_per90'] - self.df['goals_per90'].min()) / (self.df['goals_per90'].max() - self.df['goals_per90'].min())
        assists_norm = (self.df['assists_per90'] - self.df['assists_per90'].min()) / (self.df['assists_per90'].max() - self.df['assists_per90'].min())
        minutes_norm = (self.df['minutes_played'] - self.df['minutes_played'].min()) / (self.df['minutes_played'].max() - self.df['minutes_played'].min())
        
        # Score avec pondération (buts 40%, passes 35%, temps de jeu 25%)
        performance_score = (
            goals_norm * 0.4 +
            assists_norm * 0.35 + 
            minutes_norm * 0.25
        ) * self.df['league_weight']
        
        return performance_score
    
    def get_analysis_summary(self) -> Dict:
        """Générer un résumé analytique des données"""
        logger.info("Génération du résumé analytique...")
        
        summary = {
            'general': {
                'total_players': len(self.df),
                'total_goals': int(self.df['goals'].sum()),
                'total_assists': int(self.df['assists'].sum()),
                'total_minutes': int(self.df['minutes_played'].sum()),
                'avg_age': self.df['age'].mean(),
                'leagues_count': self.df['league'].nunique(),
                'teams_count': self.df['team'].nunique()
            },
            'by_league': self.df.groupby('league').agg({
                'player_name': 'count',
                'goals': 'sum',
                'assists': 'sum',
                'goals_per90': 'mean',
                'age': 'mean'
            }).round(2).to_dict('index'),
            'by_position': self.df['position_group'].value_counts().to_dict(),
            'top_performers': {
                'top_scorers': self.df.nlargest(5, 'goals')[['player_name', 'team', 'goals']].to_dict('records'),
                'top_assisters': self.df.nlargest(5, 'assists')[['player_name', 'team', 'assists']].to_dict('records'),
                'top_efficient': self.df.nlargest(5, 'performance_score')[['player_name', 'team', 'performance_score']].to_dict('records')
            }
        }
        
        return summary
    
    def save_processed_data(self, output_path: str = "data/players_processed.csv"):
        """Sauvegarder les données traitées"""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Données traitées sauvegardées: {output_path}")
        else:
            logger.warning("Aucune donnée à sauvegarder")

# Script de test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 DÉBUT DU TRAITEMENT DES DONNÉES")
    print("=" * 50)
    
    try:
        processor = DataProcessor()
        
        # Charger et traiter les données
        processed_df = processor.load_and_clean_data()
        
        # Générer le résumé
        summary = processor.get_analysis_summary()
        
        # Sauvegarder
        processor.save_processed_data()
        
        print(f"\n✅ TRAITEMENT TERMINÉ:")
        print(f"   • {summary['general']['total_players']} joueurs traités")
        print(f"   • {summary['general']['total_goals']} buts au total")
        print(f"   • {summary['general']['total_assists']} passes décisives")
        print(f"   • {summary['general']['leagues_count']} ligues analysées")
        
        print(f"\n🏆 MEILLEURS BUTEURS:")
        for player in summary['top_performers']['top_scorers']:
            print(f"   • {player['player_name']} ({player['team']}): {player['goals']} buts")
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()