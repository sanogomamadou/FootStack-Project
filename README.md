# 🚀 FootStack - Plateforme Complète d'Intelligence Footballistique

```mermaid
graph TB
    subgraph "Data Sources"
        A[Football-Data.org API]
        B[FBref Web Scraping]
    end
    
    subgraph "Data Ingestion"
        C[API Client]
        D[Selenium Scraper]
        E[Data Validation]
    end
    
    subgraph "Storage"
        F[PostgreSQL]
        G[Models Storage]
    end
    
    subgraph "ML Pipeline"
        H[Feature Engineering]
        I[Model Training]
        J[Clustering]
    end
    
    subgraph "API Layer"
        K[FastAPI Server]
        L[Prediction Routes]
        M[Talent Scout Routes]
    end
    
    subgraph "Orchestration"
        N[Airflow Scheduler]
        O[DAG Management]
    end
    
    
    A --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> H
    H --> I
    H --> J
    I --> G
    J --> G
    G --> L
    G --> M
    F --> L
    F --> M
    L --> K
    M --> K
    N --> C
    N --> D
    N --> I
    N --> J
```

## 📖 Table des Matières
- [🎯 Aperçu du Projet](#-aperçu-du-projet)
- [🏗️ Architecture du Système](#️-architecture-du-système)
- [🛠️ Stack Technologique](#️-stack-technologique)
- [📊 Fonctionnalités](#-fonctionnalités)
- [🚀 Installation et Démarrage](#-installation-et-démarrage)
- [🔧 Modules Détaillés](#-modules-détaillés)
- [📈 Résultats et Performances](#-résultats-et-performances)
- [💡 Compétences Développées](#-compétences-développées)
- [🔮 Améliorations Futures](#-améliorations-futures)

## 🎯 Aperçu du Projet

**FootStack** est une plateforme end-to-end d'analyse footballistique qui combine **Machine Learning**, **Data Engineering** et **Software Development** pour fournir des prédictions de matchs et de la détection de talents façon "Moneyball".

### 🎯 Objectifs Principaux
- ✅ **Prédire les résultats de matchs** avec des modèles ML avancés
- ✅ **Identifier des joueurs sous-évalués** via clustering et analyse statistique  
- ✅ **Automatiser les pipelines de données** avec Airflow
- ✅ **Exposer les fonctionnalités** via une API REST professionnelle
- ✅ **Conteneuriser l'application** pour un déploiement facile


## 🏗️ Architecture du Système

```mermaid
graph TB
    A[Sources de Données] --> B[Data Ingestion]
    B --> C[PostgreSQL]
    C --> D[Feature Engineering]
    D --> E[ML Pipeline]
    E --> F[FastAPI]
    C --> G[Clustering]
    G --> H[Talent Scout]
    H --> I
    J[Airflow] --> B
    J --> E
    J --> G
```

### 📁 Structure du Projet
```
FootStack/
├── 🗃️ data_ingest/          # Collecte et stockage des données
├── 🤖 ml_pipeline/          # Pipeline de ML pour prédictions
├── 🔍 talent_scout/         # Détection de talents
├── 🚀 api/                  # API FastAPI
├── ⚙️ airflow/              # Orchestration des workflows
├── 📊 data/                 # Datasets et fichiers intermédiaires
├── 🧠 models/               # Modèles entraînés
└── 🐳 docker-compose.yml    # Configuration Docker
```

## 🛠️ Stack Technologique

### 🐍 Backend & Data Science
- **Python 3.9+** - Langage principal
- **FastAPI** - Framework API moderne avec documentation automatique
- **SQLAlchemy** - ORM pour la gestion de base de données
- **Pandas/NumPy** - Manipulation de données
- **Scikit-learn** - Algorithmes de ML traditionnels
- **XGBoost** - Algorithmes de boosting avancés
- **Joblib** - Sérialisation des modèles

### 🗄️ Base de Données
- **PostgreSQL 15** - Base de données relationnelle
- **Modèles relationnels** : Matchs, Équipes, Joueurs, Statistiques

### 🔄 Orchestration & Conteneurisation
- **Apache Airflow 2.7** - Orchestration des pipelines
- **Docker & Docker Compose** - Conteneurisation des services
- **Multi-service architecture** : API, DB, Airflow

### 🌐 Data Sources & Web
- **Football-Data.org API** - Données des matchs en temps réel
- **FBref** - Statistiques détaillées des joueurs (web scraping)
- **Selenium** - Automatisation du scraping

### 📊 Visualisation & Analyse
- **Matplotlib/Seaborn** - Visualisations des données
- **Scikit-learn Metrics** - Évaluation des modèles
- **Clustering K-means** - regroupement des joueurs

## 📊 Fonctionnalités

### 🎯 Module 1 - Prédiction des Matchs
![Prédictions API](images/Interface_API_Swagger.jpeg)

#### 🔮 Prédictions en Temps Réel
- **Endpoint** : `POST /predictions/predict-auto`
- **Entrée** : Équipe domicile, équipe extérieure
- **Sortie** : Probabilités (Victoire/Nul/Défaite) + Confiance
- **Features Automatiques** : Forme récente, confrontations directes, contexte

#### 📈 Features Engineering Avancé
- **Forme des équipes** (5 derniers matchs)
- **Statistiques H2H** (historique des confrontations)
- **Contexte** : jours de repos, importance du matchday
- **Performance offensive/défensive** (moyennes glissantes)

#### 🤖 Modèles de Machine Learning
- **XGBoost Optimisé** - Accuracy: 49.73%
- **Random Forest** - Performance de comparaison
- **Validation temporelle** - Split chronologique des données
- **Feature Importance** - Interprétabilité des prédictions

### 🔍 Module 2 - Détection de Talents "Moneyball"
![Clustering Analysis](images/clustering_analysis.png)

#### 🎯 Identification des Joueurs Sous-évalués
- **Algorithme** : K-means Clustering avec optimisation automatique
- **Métriques** : Goals/90, Assists/90, Contribution offensive
- **Score de sous-évaluation** : Combinaison performance + atypicité

#### 📊 Clustering Intelligent
- **Détermination automatique** du nombre optimal de clusters
- **Analyse silhouette** + méthode du coude
- **8 clusters** identifiés automatiquement
- **Profils types** : Buteurs prolifiques, Créateurs de jeu, Polyvalents

#### 🌐 API Talent Scout Complète
![Prédictions API](images/Interface_API_talent.jpeg)

```bash
# Joueurs sous-évalués
GET /talent-scout/undervalued?limit=15&min_score=0.6

# Analyse des clusters  
GET /talent-scout/clusters

# Recherche avancée
GET /talent-scout/players/search?position=Forward&min_goals=0.3

# Détails joueur + similaires
GET /talent-scout/players/Messi
```

### ⚙️ Module 3 - Orchestration Airflow
![Airflow DAG](images/DAG_Airflow.jpeg)

#### 🔄 Pipeline Automatisé
- **Collecte données** - Tous les 14 jours
- **Nettoyage** - Transformation et feature engineering
- **Entraînement** - Re-entraînement des modèles

#### 🛠️ Tâches Spécialisées
```python
# Structure du DAG
start → wait_for_db → ingest_data → clean_data 
       → engineer_features → train_models → optimize_models → end
```

### 🚀 Module 4 - API FastAPI

#### 📡 Endpoints Complets
- **Health Checks** - Monitoring de l'état du système
- **Prédictions** - Automatiques et manuelles
- **Talent Scout** - Recherche et analyse de joueurs
- **Documentation Interactive** - Swagger UI automatique

#### 🔒 Features API
- **Validation Pydantic** - Schémas stricts
- **Dependency Injection** - Gestion des dépendances
- **Error Handling** - Gestion robuste des erreurs
- **CORS** - Prêt pour l'intégration frontend
- **Logging** - Traçabilité complète

## 🚀 Installation et Démarrage

### 📋 Pré-requis
- Docker et Docker Compose
- Clé API Football-Data.org

### 🐳 Démarrage Rapide
```bash
# 1. Cloner le repository
git clone https://github.com/ton-username/footstack.git
cd footstack

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec votre clé API

# 3. Lancer l'application
docker-compose up -d

# 4. Accéder aux services
# API: http://localhost:8000
# Documentation: http://localhost:8000/docs
# Airflow: http://localhost:8080 (airflow/airflow)
```

### 🔧 Configuration Détaillée

#### Variables d'Environnement
```env
# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/footstack

# API Football Data
API_TOKEN=ta_cle_api_football_data
API_BASE=https://api.football-data.org/v4/

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
```

#### Services Docker
```yaml
services:
  postgres:           # Base de données principale
  airflow-postgres:   # Métadata Airflow  
  airflow-webserver:  # Interface Airflow
  airflow-scheduler:  # Planificateur Airflow


## 🔧 Modules Détaillés

### 🗃️ Data Ingestion & Storage

#### Sources de Données
- **Football-Data.org** : Matchs en temps réel, classements
- **FBref** : Statistiques détaillées des joueurs (scraping)
- **Périmètre** : Top 5 leagues européennes (2020-2024)

#### Pipeline de Collecte
```python
# Workflow complet
1. Authentification API Football-Data
2. Collecte compétitions (Premier League, La Liga, etc.)
3. Récupération matchs historiques (2 ans)
4. Scraping FBref pour statistiques joueurs
5. Nettoyage et standardisation
6. Stockage PostgreSQL
```

### 🤖 Machine Learning Pipeline

#### Feature Engineering
```python
features = {
    # Forme récente (5 derniers matchs)
    'home_points_avg_5', 'home_goals_for_avg_5', 'home_goals_against_avg_5',
    
    # Confrontations directes  
    'h2h_win_rate_home', 'h2h_win_rate_away', 'h2h_matches_played',
    
    # Contexte match
    'home_days_rest', 'away_days_rest', 'rest_advantage', 'matchday_importance'
}
```

#### Modèles Implémentés
- **XGBoost Optimisé** : Meilleure performance
- **Random Forest** : Baseline robuste
- **Optimisation** : GridSearchCV pour hyperparamètres

#### Évaluation
- **Split temporel** : 80/20 chronologique
- **Métriques** : Accuracy, Precision, Recall, F1-Score
- **Feature Importance** : Analyse de l'impact des variables

### 🔍 Talent Scout & Clustering

#### Algorithme de Clustering
```python
# Pipeline de clustering
1. Préparation features (goals_per90, assists_per90, etc.)
2. Standardisation des données
3. Détermination k optimal (méthode coude)
4. Application K-means
5. Analyse des clusters
6. Identification joueurs sous-évalués


## 💡 Compétences Développées

### 🔧 Data Engineering
- ✅ **Architecture ETL/ELT** - Pipeline de données end-to-end
- ✅ **API Integration** - Consommation REST APIs avec rate limiting
- ✅ **Web Scraping** - Extraction données structurées (Selenium)
- ✅ **Modélisation BD** - Conception schémas relationnels complexes
- ✅ **SQL Avancé** - Requêtes complexes, optimisations
- ✅ **Data Quality** - Validation, nettoyage, standardisation

### 🤖 Machine Learning
- ✅ **Feature Engineering** - Création features métier pertinentes
- ✅ **Modèles Supervisés** - XGBoost, Random Forest, optimisation
- ✅ **Clustering** - K-means, détermination k optimal, analyse
- ✅ **Validation** - Split temporel, cross-validation, métriques
- ✅ **MLOps** - Versioning modèles, pipelines reproductibles
- ✅ **Interpretabilité** - Feature importance, analyse décisions

### 🚀 Software Engineering
- ✅ **API Design** - Architecture REST, documentation OpenAPI
- ✅ **FastAPI** - Framework moderne, async, validation
- ✅ **Docker** - Conteneurisation multi-service
- ✅ **Orchestration** - Airflow, DAGs, gestion dépendances
- ✅ **Architecture Microservices** - Services découplés
- ✅ **Testing** - Tests d'intégration, health checks

### 📊 Data Science Avancée
- ✅ **Time Series Analysis** - Données chronologiques matchs
- ✅ **Statistical Modeling** - Approche "Moneyball", valeur ajoutée
- ✅ **Data Visualization** - Analyse résultats, clustering
- ✅ **Business Insight** - Translation technique → valeur métier

### 🔄 DevOps & Production
- ✅ **CI/CD Ready** - Architecture prête déploiement
- ✅ **Monitoring** - Logging, health checks, métriques
- ✅ **Performance** - Optimisation requêtes, caching
- ✅ **Scalability** - Architecture horizontale possible
- ✅ **Documentation** - Documentation technique complète

## 🔮 Améliorations Futures

### 🎯 Court Terme
- [ ] **Dashboard React** - Interface utilisateur complète
- [ ] **Monitoring** - Métriques détaillées avec Prometheus

### 🚀 Moyen Terme  
- [ ] **Module Computer Vision** - Analyse vidéo des matchs
- [ ] **Module LLM** - Assistant footballistique conversationnel

### 🔮 Long Terme
- [ ] **Mobile App** - Application mobile predictions
- [ ] **Social Features** - Communauté, pronostics
- [ ] **Advanced Analytics** - xG, pressing indexes, etc.


---

**FootStack** - *Révolutionnez l'analyse footballistique avec l'IA* ⚽🎯

*Développé avec passion pour le football et la data science*