# talent_scout/modeling/clustering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import joblib
from datetime import datetime
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

logger = logging.getLogger(__name__)

# Imports pour la base de données
try:
    from data_ingest.db import SessionLocal
    from data_ingest.models import PlayerCluster
    print("✅ Modules data_ingest importés avec succès")
except ImportError as e:
    print(f"❌ Erreur import data_ingest: {e}")
    sys.exit(1)

class PlayerClustering:
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Préparer les features pour le clustering
        """
        logger.info("Préparation des features pour le clustering...")
        
        # Features de base pour tous les joueurs (sauf gardiens)
        base_features = [
            'goals_per90', 'assists_per90', 'goal_contrib_per90', 
            'minutes_played_norm'
        ]
        
        # S'assurer que toutes les colonnes existent
        available_features = [col for col in base_features if col in df.columns]
        
        if len(available_features) < 2:
            raise ValueError("Pas assez de features disponibles pour le clustering")
            
        features_df = df[available_features].copy()
        
        # Nettoyer les valeurs infinies et NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Filtrer les outliers extrêmes (top 1%)
        for col in available_features:
            q99 = features_df[col].quantile(0.99)
            features_df[col] = np.where(features_df[col] > q99, q99, features_df[col])
        
        self.feature_columns = available_features
        logger.info(f"Features utilisées: {self.feature_columns}")
        
        return features_df
    
    def find_optimal_clusters(self, df: pd.DataFrame, max_k: int = 15) -> int:
        """
        Trouver le nombre optimal de clusters avec la méthode du coude
        """
        logger.info("Recherche du nombre optimal de clusters...")
        
        features_df = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(features_df)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            
            # Score de silhouette (éviter quand k=1)
            if k > 1:
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        # Méthode du coude simplifiée
        optimal_k = 8  # Valeur par défaut pour commencer
        
        # Trouver le "coude" dans la courbe d'inertie
        differences = []
        for i in range(1, len(inertias)):
            diff = inertias[i-1] - inertias[i]
            differences.append(diff)
        
        if differences:
            # Le coude est où la réduction d'inertie diminue significativement
            avg_reduction = np.mean(differences)
            for i, diff in enumerate(differences):
                if diff < avg_reduction * 0.5:  # Quand la réduction chute de moitié
                    optimal_k = i + 2  # +2 car on commence à k=2
                    break
        
        logger.info(f"Nombre optimal de clusters suggéré: {optimal_k}")
        
        # Visualisation (optionnelle)
        self._plot_elbow_curve(k_range, inertias, silhouette_scores, optimal_k)
        
        return optimal_k
    
    def _plot_elbow_curve(self, k_range, inertias, silhouette_scores, optimal_k):
        """Visualiser la courbe du coude"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Courbe d'inertie
            ax1.plot(k_range, inertias, 'bo-')
            ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Nombre de clusters')
            ax1.set_ylabel('Inertie')
            ax1.set_title('Méthode du Coude')
            ax1.grid(True, alpha=0.3)
            
            # Score de silhouette
            ax2.plot(k_range[1:], silhouette_scores[1:], 'go-')
            ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Nombre de clusters')
            ax2.set_ylabel('Score Silhouette')
            ax2.set_title('Score de Silhouette')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('data/clustering_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📊 Graphique d'analyse sauvegardé: data/clustering_analysis.png")
            
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder le graphique: {e}")
    
    def fit_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appliquer le clustering K-means
        """
        logger.info("Application du clustering K-means...")
        
        # Préparer les features
        features_df = self.prepare_features(df)
        
        # Standardiser les features
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Ajuster le modèle K-means
        self.kmeans.fit(X_scaled)
        
        # Ajouter les clusters au DataFrame original
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = self.kmeans.labels_
        
        # Calculer la distance au centroïde (mesure de typicalité)
        distances = self.kmeans.transform(X_scaled)
        df_with_clusters['distance_to_centroid'] = distances.min(axis=1)
        
        # Analyser les clusters
        cluster_analysis = self._analyze_clusters(df_with_clusters)
        
        logger.info(f"✅ Clustering terminé. {self.n_clusters} clusters créés.")
        logger.info(f"📈 Distribution: {df_with_clusters['cluster'].value_counts().to_dict()}")
        
        return df_with_clusters, cluster_analysis
    
    def _analyze_clusters(self, df: pd.DataFrame) -> Dict:
        """
        Analyser les caractéristiques de chaque cluster
        """
        analysis = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Caractéristiques moyennes du cluster
            cluster_profile = {
                'size': len(cluster_data),
                'avg_goals_per90': cluster_data['goals_per90'].mean(),
                'avg_assists_per90': cluster_data['assists_per90'].mean(),
                'avg_goal_contrib_per90': cluster_data['goal_contrib_per90'].mean(),
                'avg_minutes_played': cluster_data['minutes_played'].mean(),
                'main_positions': cluster_data['position_group'].value_counts().head(3).to_dict(),
                'top_teams': cluster_data['team'].value_counts().head(3).to_dict()
            }
            
            analysis[cluster_id] = cluster_profile
        
        return analysis
    
    def find_undervalued_players(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Identifier les joueurs sous-évalués (haute performance mais dans des petits clusters)
        """
        logger.info("Recherche de joueurs sous-évalués...")
        
        # Calculer un score de performance simple
        df['performance_score'] = (
            df['goals_per90'] * 0.6 + 
            df['assists_per90'] * 0.4 +
            df['goal_contrib_per90'] * 0.3
        )
        
        # Normaliser le score de performance
        df['performance_score_norm'] = (
            (df['performance_score'] - df['performance_score'].min()) / 
            (df['performance_score'].max() - df['performance_score'].min())
        )
        
        # Identifier les clusters avec haute performance moyenne
        cluster_performance = df.groupby('cluster')['performance_score_norm'].mean()
        high_perf_clusters = cluster_performance[cluster_performance > cluster_performance.median()].index
        
        # Trouver les joueurs performants dans des petits clusters (potentiellement sous-évalués)
        undervalued_candidates = df[
            (df['cluster'].isin(high_perf_clusters)) &
            (df['performance_score_norm'] > 0.7)  # Hautement performants
        ].copy()
        
        # Trier par performance et distance au centroïde (plus atypique = plus intéressant)
        undervalued_candidates['undervalued_score'] = (
            undervalued_candidates['performance_score_norm'] * 0.7 +
            (1 - undervalued_candidates['distance_to_centroid']) * 0.3  # Plus proche du centroïde = plus typique
        )
        
        top_undervalued = undervalued_candidates.nlargest(top_n, 'undervalued_score')[
            ['name', 'team', 'position_group', 'goals_per90', 'assists_per90', 
             'performance_score_norm', 'cluster', 'distance_to_centroid', 'undervalued_score']
        ]
        
        logger.info(f"🎯 {len(top_undervalued)} joueurs sous-évalués identifiés")
        
        return top_undervalued
    
    def save_model(self, filepath: str = "models/player_clustering.joblib"):
        """Sauvegarder le modèle de clustering"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_clusters': self.n_clusters,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Modèle sauvegardé: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = "models/player_clustering.joblib"):
        """Charger un modèle de clustering sauvegardé"""
        model_data = joblib.load(filepath)
        
        instance = cls(n_clusters=model_data['n_clusters'])
        instance.kmeans = model_data['kmeans']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        
        logger.info(f"📂 Modèle chargé: {filepath}")
        return instance

    def save_clusters_to_database(self, df_clustered: pd.DataFrame):
        """
        Sauvegarder les résultats du clustering dans la table player_clusters
        """
        logger.info("💾 Sauvegarde des clusters en base de données...")
        
        session = SessionLocal()
        try:
            # Vider la table existante pour éviter les doublons
            session.query(PlayerCluster).delete()
            
            # Insérer les nouveaux clusters
            clusters_to_insert = []
            for _, row in df_clustered.iterrows():
                # Vérifier que l'ID du joueur existe
                if pd.notna(row.get('id_info')):
                    cluster_record = PlayerCluster(
                        player_id=int(row['id_info']),
                        cluster_id=int(row['cluster']),
                        position_group=row.get('position_group', 'Unknown'),
                        similarity_score=float(1 - row.get('distance_to_centroid', 1.0)),
                        created_at=datetime.utcnow()
                    )
                    clusters_to_insert.append(cluster_record)
            
            # Insertion en batch
            session.bulk_save_objects(clusters_to_insert)
            session.commit()
            
            logger.info(f"✅ {len(clusters_to_insert)} clusters sauvegardés en base de données")
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Erreur sauvegarde clusters en base: {e}")
            raise
        finally:
            session.close()

def run_complete_clustering_pipeline():
    """
    Exécuter le pipeline complet de clustering
    """
    logger.info("🚀 Démarrage du pipeline complet de clustering...")
    
    try:
        # MODIFICATION : Charger directement depuis le CSV existant
        logger.info("📂 Chargement des données depuis data/processed_players.csv...")
        df = pd.read_csv("data/processed_players.csv")
        logger.info(f"✅ Données chargées: {len(df)} joueurs")
        
    except FileNotFoundError:
        # Fallback : utiliser le data_processor si le CSV n'existe pas
        logger.info("📂 Fichier CSV non trouvé, utilisation du data_processor...")
        try:
            from talent_scout.data_collection.data_processor import process_player_data
            df = process_player_data(save_csv=True)
        except ImportError as e:
            logger.error(f"❌ Impossible d'importer data_processor: {e}")
            logger.error("💡 Assure-toi que le fichier data/processed_players.csv existe")
            return None, None, None
    
    if df.empty:
        logger.error("❌ Aucune donnée à clusteriser")
        return None, None, None
    
    # 2. Initialiser et configurer le clustering
    cluster_analyzer = PlayerClustering(n_clusters=8)
    
    # 3. Trouver le nombre optimal de clusters
    optimal_k = cluster_analyzer.find_optimal_clusters(df)
    cluster_analyzer.n_clusters = optimal_k
    
    # 4. Appliquer le clustering
    df_clustered, cluster_analysis = cluster_analyzer.fit_clusters(df)
    
    # 5. Identifier les joueurs sous-évalués
    undervalued_players = cluster_analyzer.find_undervalued_players(df_clustered, top_n=15)
    
    # 6. Sauvegarder les résultats
    df_clustered.to_csv("data/players_with_clusters.csv", index=False)
    cluster_analyzer.save_model()
    
    # 7. NOUVEAU : Sauvegarder en base de données
    cluster_analyzer.save_clusters_to_database(df_clustered)
    
    # 8. Afficher les résultats
    print("\n" + "="*70)
    print("🎯 RAPPORT DE CLUSTERING - JOUEURS SOUS-ÉVALUÉS")
    print("="*70)
    
    print(f"\n🏆 TOP 15 JOUEURS SOUS-ÉVALUÉS:")
    for idx, (_, player) in enumerate(undervalued_players.iterrows(), 1):
        print(f"{idx:2d}. {player['name']:25} | {player['team']:20} | "
              f"{player['position_group']:10} | Buts/90: {player['goals_per90']:4.2f} | "
              f"Passes/90: {player['assists_per90']:4.2f} | Score: {player['undervalued_score']:.3f}")
    
    print(f"\n📊 ANALYSE DES {optimal_k} CLUSTERS:")
    for cluster_id, profile in cluster_analysis.items():
        main_positions = list(profile['main_positions'].keys())[:2] if profile['main_positions'] else []
        positions_str = ", ".join(main_positions)
        print(f"   Cluster {cluster_id}: {profile['size']:3} joueurs | "
              f"Buts/90: {profile['avg_goals_per90']:5.2f} | "
              f"Passes/90: {profile['avg_assists_per90']:5.2f} | "
              f"Positions: {positions_str}")
    
    print(f"\n📈 STATISTIQUES GLOBALES:")
    print(f"   • Total joueurs clusterisés: {len(df_clustered)}")
    print(f"   • Joueurs sous-évalués identifiés: {len(undervalued_players)}")
    print(f"   • Score moyen des sous-évalués: {undervalued_players['undervalued_score'].mean():.3f}")
    
    # Vérification base de données
    try:
        session = SessionLocal()
        cluster_count = session.query(PlayerCluster).count()
        session.close()
        print(f"   • Clusters en base de données: {cluster_count}")
    except Exception as e:
        print(f"   • ❌ Erreur vérification base: {e}")
    
    print(f"\n💡 INTERPRÉTATION:")
    print("   • Score > 0.8: Très sous-évalué - forte recommandation")
    print("   • Score 0.6-0.8: Sous-évalué - bonne opportunité") 
    print("   • Score < 0.6: Potentiel à surveiller")
    
    return df_clustered, undervalued_players, cluster_analysis

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🔧 Initialisation du clustering des joueurs...")
    df_clustered, undervalued_players, cluster_analysis = run_complete_clustering_pipeline()
    
    if df_clustered is not None:
        print("\n✅ Clustering terminé avec succès!")
        print("💾 Fichiers créés:")
        print("   - data/players_with_clusters.csv")
        print("   - data/clustering_analysis.png") 
        print("   - models/player_clustering.joblib")
    else:
        print("\n❌ Le clustering a échoué")