# api/services/talent_scout.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class TalentScoutService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.clustering_model = None
        self._load_clustering_model()
    
    def _load_clustering_model(self):
        """Charger le modèle de clustering"""
        try:
            model_path = "models/player_clustering.joblib"
            self.clustering_model = joblib.load(model_path)
            logger.info("✅ Modèle de clustering chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle clustering: {e}")
            self.clustering_model = None
    
    def get_undervalued_players(self, top_n: int = 15) -> Dict:
        """
        Récupérer les joueurs sous-évalués depuis la base de données
        """
        try:
            logger.info(f"🔍 Recherche des {top_n} joueurs sous-évalués...")
            
            # Requête pour récupérer les joueurs sous-évalués avec leurs clusters
            query = text("""
            SELECT 
                p.name,
                p.team,
                p.position as position_group,
                (ps.data->>'goals_per90')::float as goals_per90,
                (ps.data->>'assists_per90')::float as assists_per90,
                pc.cluster_id as cluster,
                pc.similarity_score,
                (ps.data->>'performance_score_norm')::float as performance_score_norm,
                (1 - pc.similarity_score) as distance_to_centroid,
                (ps.data->>'undervalued_score')::float as undervalued_score
            FROM player_clusters pc
            JOIN players p ON pc.player_id = p.id
            JOIN player_stats ps ON p.id = ps.player_id
            WHERE (ps.data->>'undervalued_score')::float > 0.6
            ORDER BY (ps.data->>'undervalued_score')::float DESC
            LIMIT :limit
            """)
            
            players_df = pd.read_sql_query(query, self.db.bind, params={"limit": top_n})
            
            # Analyse des clusters
            cluster_query = text("""
            SELECT 
                cluster_id,
                COUNT(*) as size,
                AVG(1 - similarity_score) as avg_distance,
                position_group
            FROM player_clusters 
            GROUP BY cluster_id, position_group
            ORDER BY cluster_id
            """)
            
            clusters_df = pd.read_sql_query(cluster_query, self.db.bind)
            
            # Préparer l'analyse des clusters
            cluster_analysis = {}
            for cluster_id in clusters_df['cluster_id'].unique():
                cluster_data = clusters_df[clusters_df['cluster_id'] == cluster_id]
                position_counts = cluster_data.set_index('position_group')['size'].to_dict()
                
                cluster_analysis[cluster_id] = {
                    'size': int(cluster_data['size'].sum()),
                    'avg_distance': float(cluster_data['avg_distance'].mean()),
                    'main_positions': position_counts
                }
            
            logger.info(f"✅ {len(players_df)} joueurs sous-évalués trouvés")
            
            return {
                'undervalued_players': players_df.to_dict('records'),
                'cluster_analysis': cluster_analysis,
                'total_players': len(players_df),
                'optimal_clusters': len(cluster_analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche joueurs sous-évalués: {e}")
            return {
                'undervalued_players': [],
                'cluster_analysis': {},
                'total_players': 0,
                'optimal_clusters': 0
            }
    
    def search_players_by_profile(self, request) -> List[Dict]:
        """
        Rechercher des joueurs par profil spécifique
        """
        try:
            logger.info("🔍 Recherche de joueurs par profil...")
            
            query_params = {}
            where_conditions = []
            
            # Construire la requête dynamiquement
            if request.position:
                where_conditions.append("p.position LIKE :position")
                query_params["position"] = f"%{request.position}%"
            
            if request.team:
                where_conditions.append("p.team = :team")
                query_params["team"] = request.team
            
            if request.min_goals_per90 > 0:
                where_conditions.append("(ps.data->>'goals_per90')::float >= :min_goals")
                query_params["min_goals"] = request.min_goals_per90
            
            if request.min_assists_per90 > 0:
                where_conditions.append("(ps.data->>'assists_per90')::float >= :min_assists")
                query_params["min_assists"] = request.min_assists_per90
            
            # Requête de base
            base_query = """
            SELECT 
                p.name,
                p.team,
                p.position as position_group,
                p.age,
                p.nationality,
                (ps.data->>'goals_per90')::float as goals_per90,
                (ps.data->>'assists_per90')::float as assists_per90,
                (ps.data->>'goal_contrib_per90')::float as goal_contrib_per90,
                ps.minutes_played,
                pc.cluster_id,
                pc.similarity_score,
                (ps.data->>'performance_score_norm')::float as performance_score_norm
            FROM players p
            JOIN player_stats ps ON p.id = ps.player_id
            LEFT JOIN player_clusters pc ON p.id = pc.player_id
            """
            
            # Ajouter les conditions WHERE
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            # Ajouter le tri et la limite
            base_query += " ORDER BY (ps.data->>'performance_score_norm')::float DESC LIMIT :limit"
            query_params["limit"] = request.limit
            
            players_df = pd.read_sql_query(text(base_query), self.db.bind, params=query_params)
            
            logger.info(f"✅ {len(players_df)} joueurs trouvés avec le profil spécifié")
            
            return players_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche par profil: {e}")
            return []
    
    def find_similar_players(self, player_name: str, max_results: int = 10) -> List[Dict]:
        """
        Trouver des joueurs similaires à un joueur donné
        """
        try:
            logger.info(f"🔍 Recherche de joueurs similaires à {player_name}...")
            
            # Trouver le cluster du joueur cible
            cluster_query = text("""
            SELECT pc.cluster_id, pc.similarity_score
            FROM player_clusters pc
            JOIN players p ON pc.player_id = p.id
            WHERE p.name ILIKE :player_name
            LIMIT 1
            """)
            
            target_cluster = pd.read_sql_query(
                cluster_query, 
                self.db.bind, 
                params={"player_name": f"%{player_name}%"}
            )
            
            if target_cluster.empty:
                logger.warning(f"❌ Joueur {player_name} non trouvé")
                return []
            
            cluster_id = target_cluster['cluster_id'].iloc[0]
            target_similarity = target_cluster['similarity_score'].iloc[0]
            
            # Trouver des joueurs dans le même cluster
            similar_query = text("""
            SELECT 
                p.name,
                p.team,
                p.position as position_group,
                (ps.data->>'goals_per90')::float as goals_per90,
                (ps.data->>'assists_per90')::float as assists_per90,
                pc.similarity_score,
                ABS(pc.similarity_score - :target_similarity) as similarity_diff
            FROM player_clusters pc
            JOIN players p ON pc.player_id = p.id
            JOIN player_stats ps ON p.id = ps.player_id
            WHERE pc.cluster_id = :cluster_id 
            AND p.name NOT ILIKE :player_name
            ORDER BY similarity_diff ASC, pc.similarity_score DESC
            LIMIT :max_results
            """)
            
            similar_players = pd.read_sql_query(
                similar_query,
                self.db.bind,
                params={
                    "cluster_id": cluster_id,
                    "target_similarity": target_similarity,
                    "player_name": f"%{player_name}%",
                    "max_results": max_results
                }
            )
            
            logger.info(f"✅ {len(similar_players)} joueurs similaires trouvés")
            
            return similar_players.to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche joueurs similaires: {e}")
            return []
    
    def get_cluster_analysis(self) -> Dict:
        """
        Obtenir une analyse détaillée de tous les clusters
        """
        try:
            logger.info("📊 Génération de l'analyse des clusters...")
            
            query = text("""
            SELECT 
                pc.cluster_id,
                COUNT(*) as cluster_size,
                p.position as position_group,
                p.team,
                AVG((ps.data->>'goals_per90')::float) as avg_goals_per90,
                AVG((ps.data->>'assists_per90')::float) as avg_assists_per90,
                AVG((ps.data->>'goal_contrib_per90')::float) as avg_goal_contrib_per90,
                AVG(pc.similarity_score) as avg_similarity
            FROM player_clusters pc
            JOIN players p ON pc.player_id = p.id
            JOIN player_stats ps ON p.id = ps.player_id
            GROUP BY pc.cluster_id, p.position, p.team
            ORDER BY pc.cluster_id, cluster_size DESC
            """)
            
            clusters_df = pd.read_sql_query(query, self.db.bind)
            
            # Structurer l'analyse
            cluster_analysis = {}
            for cluster_id in clusters_df['cluster_id'].unique():
                cluster_data = clusters_df[clusters_df['cluster_id'] == cluster_id]
                
                # Positions principales
                position_counts = cluster_data.groupby('position_group')['cluster_size'].sum().nlargest(3).to_dict()
                
                # Équipes principales
                team_counts = cluster_data.groupby('team')['cluster_size'].sum().nlargest(3).to_dict()
                
                cluster_analysis[cluster_id] = {
                    'size': int(cluster_data['cluster_size'].sum()),
                    'avg_goals_per90': float(cluster_data['avg_goals_per90'].mean()),
                    'avg_assists_per90': float(cluster_data['avg_assists_per90'].mean()),
                    'avg_goal_contrib_per90': float(cluster_data['avg_goal_contrib_per90'].mean()),
                    'avg_similarity': float(cluster_data['avg_similarity'].mean()),
                    'main_positions': position_counts,
                    'top_teams': team_counts
                }
            
            logger.info(f"✅ Analyse de {len(cluster_analysis)} clusters générée")
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse clusters: {e}")
            return {}