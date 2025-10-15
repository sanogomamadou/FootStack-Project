# api/services/talent_scout_service.py
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class TalentScoutService:
    def __init__(self):
        self.clustering_model = None
        self.df_clustered = None
        
    def load_data(self):
        """Charger les données clusterisées"""
        try:
            self.df_clustered = pd.read_csv("data/players_with_clusters.csv")
            logger.info(f"✅ Données chargées: {len(self.df_clustered)} joueurs")
            
            # Convertir les types NumPy en types Python natifs
            self._convert_numpy_types()
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données: {e}")
            self.df_clustered = None
    
    def _convert_numpy_types(self):
        """Convertir les types NumPy en types Python natifs pour la sérialisation JSON"""
        if self.df_clustered is not None:
            for col in self.df_clustered.columns:
                if self.df_clustered[col].dtype == np.int64:
                    self.df_clustered[col] = self.df_clustered[col].astype(int)
                elif self.df_clustered[col].dtype == np.float64:
                    self.df_clustered[col] = self.df_clustered[col].astype(float)
    
    def get_undervalued_players(self, limit: int = 20, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Obtenir les joueurs sous-évalués"""
        if self.df_clustered is None:
            self.load_data()
        
        if self.df_clustered is None or self.df_clustered.empty:
            return []
        
        try:
            # Vérifier si la colonne existe, sinon la calculer
            if 'undervalued_score' not in self.df_clustered.columns:
                self.df_clustered = self._calculate_undervalued_score(self.df_clustered)
            
            # BAISSER le seuil pour trouver plus de joueurs
            undervalued = self.df_clustered[
                self.df_clustered['undervalued_score'] >= min_score
            ].nlargest(limit, 'undervalued_score')
            
            players = []
            for _, player in undervalued.iterrows():
                player_data = {
                    'player_name': str(player.get('name', 'Unknown')),
                    'team': str(player.get('team', 'Unknown')),
                    'position_group': str(player.get('position_group', 'Unknown')),
                    'goals_per90': float(player.get('goals_per90', 0)),
                    'assists_per90': float(player.get('assists_per90', 0)),
                    'goal_contrib_per90': float(player.get('goal_contrib_per90', 0)),
                    'performance_score': float(player.get('performance_score_norm', 0)),
                    'cluster_id': int(player.get('cluster', 0)),
                    'undervalued_score': float(player.get('undervalued_score', 0)),
                    'minutes_played': int(player.get('minutes_played', 0))
                }
                players.append(player_data)
            
            logger.info(f"🎯 {len(players)} joueurs sous-évalués trouvés (score >= {min_score})")
            return players
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération joueurs sous-évalués: {e}")
            return []
    
    def get_cluster_analysis(self) -> List[Dict[str, Any]]:
        """Obtenir l'analyse de tous les clusters"""
        if self.df_clustered is None:
            self.load_data()
        
        if self.df_clustered is None or self.df_clustered.empty:
            return []
        
        try:
            clusters_analysis = []
            for cluster_id in sorted(self.df_clustered['cluster'].unique()):
                cluster_data = self.df_clustered[self.df_clustered['cluster'] == cluster_id]
                
                # Top joueurs du cluster
                top_players = cluster_data.nlargest(5, 'goals_per90')[
                    ['name', 'team', 'position_group', 'goals_per90', 'assists_per90']
                ]
                
                # Convertir en types Python natifs
                top_players_list = []
                for _, player in top_players.iterrows():
                    top_players_list.append({
                        'player_name': str(player['name']),
                        'team': str(player['team']),
                        'position_group': str(player['position_group']),
                        'goals_per90': float(player['goals_per90']),
                        'assists_per90': float(player['assists_per90'])
                    })
                
                # Compter les positions (convertir en types natifs)
                main_positions = {}
                for pos, count in cluster_data['position_group'].value_counts().head(3).items():
                    main_positions[str(pos)] = int(count)
                
                # Compter les équipes (convertir en types natifs)
                top_teams = {}
                for team, count in cluster_data['team'].value_counts().head(3).items():
                    top_teams[str(team)] = int(count)
                
                analysis = {
                    'cluster_id': int(cluster_id),
                    'size': int(len(cluster_data)),
                    'avg_goals_per90': float(cluster_data['goals_per90'].mean()),
                    'avg_assists_per90': float(cluster_data['assists_per90'].mean()),
                    'avg_goal_contrib_per90': float(cluster_data['goal_contrib_per90'].mean()),
                    'main_positions': main_positions,
                    'top_teams': top_teams,
                    'top_players': top_players_list,
                    'description': self._generate_cluster_description(cluster_id, cluster_data)
                }
                clusters_analysis.append(analysis)
            
            logger.info(f"📊 Analyse de {len(clusters_analysis)} clusters générée")
            return clusters_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse clusters: {e}")
            return []
    
    def search_players(self, name: Optional[str] = None, team: Optional[str] = None, 
                      position: Optional[str] = None, cluster: Optional[int] = None,
                      limit: int = 20) -> List[Dict[str, Any]]:
        """Rechercher des joueurs"""
        if self.df_clustered is None:
            self.load_data()
        
        if self.df_clustered is None or self.df_clustered.empty:
            return []
        
        try:
            df_filtered = self.df_clustered.copy()
            
            # Appliquer les filtres
            if name:
                df_filtered = df_filtered[df_filtered['name'].str.contains(name, case=False, na=False)]
            if team:
                df_filtered = df_filtered[df_filtered['team'].str.contains(team, case=False, na=False)]
            if position:
                df_filtered = df_filtered[df_filtered['position_group'].str.contains(position, case=False, na=False)]
            if cluster is not None:
                df_filtered = df_filtered[df_filtered['cluster'] == cluster]
            
            # Limiter les résultats
            results = df_filtered.head(limit)
            
            players = []
            for _, player in results.iterrows():
                player_data = {
                    'player_name': str(player.get('name', 'Unknown')),
                    'team': str(player.get('team', 'Unknown')),
                    'position_group': str(player.get('position_group', 'Unknown')),
                    'goals_per90': float(player.get('goals_per90', 0)),
                    'assists_per90': float(player.get('assists_per90', 0)),
                    'goal_contrib_per90': float(player.get('goal_contrib_per90', 0)),
                    'cluster_id': int(player.get('cluster', 0)),
                    'undervalued_score': float(player.get('undervalued_score', 0)),
                    'minutes_played': int(player.get('minutes_played', 0)),
                    'nationality': str(player.get('nationality', 'Unknown'))
                }
                players.append(player_data)
            
            logger.info(f"🔍 {len(players)} joueurs trouvés pour la recherche")
            return players
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche joueurs: {e}")
            return []
    
    def get_player_details(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Obtenir les détails d'un joueur spécifique"""
        if self.df_clustered is None:
            self.load_data()
        
        if self.df_clustered is None or self.df_clustered.empty:
            return None
        
        try:
            # Recherche exacte ou partielle
            player_data = self.df_clustered[self.df_clustered['name'].str.lower() == player_name.lower()]
            if player_data.empty:
                player_data = self.df_clustered[self.df_clustered['name'].str.contains(player_name, case=False, na=False)]
            
            if player_data.empty:
                return None
            
            player = player_data.iloc[0]
            
            # Joueurs similaires (même cluster)
            similar_players = self.df_clustered[
                (self.df_clustered['cluster'] == player['cluster']) & 
                (self.df_clustered['name'] != player['name'])
            ].nlargest(5, 'goals_per90')
            
            similar = []
            for _, sim_player in similar_players.iterrows():
                similar.append({
                    'player_name': str(sim_player.get('name', 'Unknown')),
                    'team': str(sim_player.get('team', 'Unknown')),
                    'goals_per90': float(sim_player.get('goals_per90', 0)),
                    'similarity_score': 0.8
                })
            
            response = {
                'player_name': str(player.get('name', 'Unknown')),
                'team': str(player.get('team', 'Unknown')),
                'position_group': str(player.get('position_group', 'Unknown')),
                'goals_per90': float(player.get('goals_per90', 0)),
                'assists_per90': float(player.get('assists_per90', 0)),
                'goal_contrib_per90': float(player.get('goal_contrib_per90', 0)),
                'cluster_id': int(player.get('cluster', 0)),
                'undervalued_score': float(player.get('undervalued_score', 0)),
                'minutes_played': int(player.get('minutes_played', 0)),
                'nationality': str(player.get('nationality', 'Unknown')),
                'age': int(player.get('age', 0)) if pd.notna(player.get('age')) else None,
                'similar_players': similar,
                'cluster_ranking': f"Top {int((self.df_clustered[self.df_clustered['cluster'] == player['cluster']]['goals_per90'] > player['goals_per90']).sum() + 1)} dans son cluster"
            }
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erreur détails joueur: {e}")
            return None
    
    def _calculate_undervalued_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculer le score de sous-évaluation"""
        df = df.copy()
        
        # Score de performance
        df['performance_score'] = (
            df['goals_per90'] * 0.6 + 
            df['assists_per90'] * 0.4 +
            df['goal_contrib_per90'] * 0.3
        )
        
        df['performance_score_norm'] = (
            (df['performance_score'] - df['performance_score'].min()) / 
            (df['performance_score'].max() - df['performance_score'].min())
        )
        
        # Score de sous-évaluation (LOGIQUE SIMPLIFIÉE)
        cluster_performance = df.groupby('cluster')['performance_score_norm'].mean()
        
        # Tous les clusters avec performance au-dessus de la médiane
        high_perf_clusters = cluster_performance[cluster_performance > cluster_performance.median()].index
        
        df['undervalued_score'] = 0.3  # Valeur par défaut plus basse
        
        # Joueurs performants dans des clusters performants
        mask = (df['cluster'].isin(high_perf_clusters)) & (df['performance_score_norm'] > 0.5)  # Seuil plus bas
        df.loc[mask, 'undervalued_score'] = (
            df.loc[mask, 'performance_score_norm'] * 0.6 +  # Poids réduit
            (1 - df.loc[mask, 'distance_to_centroid']) * 0.4  # Poids augmenté
        )
        
        return df
    
    def _generate_cluster_description(self, cluster_id: int, cluster_data: pd.DataFrame) -> str:
        """Générer une description automatique du cluster"""
        avg_goals = cluster_data['goals_per90'].mean()
        avg_assists = cluster_data['assists_per90'].mean()
        main_position = cluster_data['position_group'].mode().iloc[0] if not cluster_data['position_group'].mode().empty else "Divers"
        
        if avg_goals > 0.4 and avg_assists > 0.2:
            return f"Attaquants prolifiques - {main_position} à haut rendement offensif"
        elif avg_goals > 0.3:
            return f"Buteurs efficaces - {main_position} avec bon taux de conversion"
        elif avg_assists > 0.25:
            return f"Créateurs de jeu - {main_position} avec forte contribution aux passes décisives"
        elif avg_goals > 0.2 and avg_assists > 0.15:
            return f"Joueurs polyvalents - {main_position} avec contribution équilibrée"
        else:
            return f"Joueurs spécialisés - {main_position} avec profil spécifique"

# Instance globale du service
talent_scout_service = TalentScoutService()