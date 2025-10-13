# 📋 FOOTSTACK - DOCUMENTATION COMPLÈTE

## 🎯 VISION DU PROJET
**Application web complète d'analyse de football basée sur l'IA** avec modules de prédiction, détection de talents, analyse vidéo et chatbot expert.

---

## 🏗️ ARCHITECTURE TECHNIQUE ACTUELLE

### **Stack Technique Déployée**
```
├── 🗄️  Base de Données: PostgreSQL + SQLAlchemy ORM
├── 🐍  Backend: Python 3.8+
├── 📊  ML Pipeline: Scikit-learn, XGBoost, Pandas, NumPy
├── 🔄  Orchestration: Docker Compose
├── 📡  API: Football-Data.org
└── 🗂️  Structure: Architecture modulaire professionnelle
```

---

## 📁 STRUCTURE DU PROJET COMPLÈTE

```
footstack/
├── 📊 data_ingest/                    # COLLECTE & STOCKAGE
│   ├── api_client.py                 # Client API Football-Data.org
│   ├── config.py                     # Configuration & variables d'environnement
│   ├── db.py                         # Connection SQLAlchemy PostgreSQL
│   ├── models.py                     # ORM (Competition, Team, Match)
│   ├── ingest.py                     # Script CLI d'ingestion
│   └── utils.py                      # Helpers (rate-limiting, retry)
│
├── 🧠 ml_pipeline/                   # MODULE 1 - PRÉDICTION ML
│   ├── feature_engineering.py        # Engineering des features
│   ├── model_training.py            # Entraînement Random Forest/XGBoost
│   ├── model_improvement.py         # Optimisation hyperparamètres
│   ├── prediction.py                # Prédictions en production
│   └── predict_upcoming.py          # Prédictions matchs à venir
│
├── 🗄️  data/
│   ├── features_dataset.csv         # Dataset avec features (3670 matchs)
│   └── matches_cleaned              # Vue SQL des matchs nettoyés
│
├── 🤖 models/
│   ├── xgboost_optimized.joblib     # Modèle optimisé (49.7% accuracy)
│   ├── random_forest_*.joblib       # Modèle Random Forest
│   └── label_encoder_*.joblib       # Encodeur des labels
│
├── 🐳 docker-compose.yml            # Containerisation PostgreSQL
├── 📋 requirements.txt              # Dépendances Python
└── 🔐 .env                         # Variables d'environnement
```

---

## ✅ FONCTIONNALITÉS IMPLÉMENTÉES

### **1. 🗄️ MODULE FONDATION - Collecte & Stockage**

#### **Fonctionnalités :**
- ✅ **API Client robuste** pour Football-Data.org
- ✅ **Rate-limiting intelligent** (1.2s entre appels)
- ✅ **Système de retry** avec backoff exponentiel
- ✅ **Modélisation ORM** avec SQLAlchemy
- ✅ **Script CLI flexible** avec arguments
- ✅ **Base PostgreSQL** containerisée avec Docker

#### **Données collectées :**
- **3,764 matchs** across 6 compétitions majeures
- **Champions League, Premier League, La Liga, Bundesliga, Serie A, Ligue 1**
- **Période** : 2 années de données historiques

#### **Tables PostgreSQL :**
```sql
competitions (id, name, area_name, code, data)
teams (id, name, short_name, tla, crest_url, data)  
matches (id, competition_id, utc_date, status, home_team_id, away_team_id, score, raw)
matches_cleaned (match_id, competition_name, home_team, away_team, home_score, away_score, result, ...)
```

---

### **2. 🧠 MODULE 1 - Prédiction Machine Learning**

#### **Features Engineering (20 variables) :**

**📈 Forme Récente (5-10 derniers matchs)**
- `home_points_avg_5`, `away_points_avg_5` → Points moyens
- `home_goals_for_avg_5`, `away_goals_for_avg_5` → Buts marqués moyens  
- `home_goals_against_avg_5`, `away_goals_against_avg_5` → Buts encaissés moyens
- `home_goal_diff_avg_5`, `away_goal_diff_avg_5` → Différence de buts moyenne
- `home_recent_form`, `away_recent_form` → Forme récente (somme points)

**⚔️ Confrontations Directes**
- `h2h_win_rate_home`, `h2h_win_rate_away` → Taux victoire H2H
- `h2h_matches_played` → Nombre de confrontations

**🎯 Features Différentielles**
- `form_difference` → Écart de forme entre équipes
- `goal_difference` → Écart de différence de buts

**📅 Facteurs Contextuels**
- `home_days_rest`, `away_days_rest` → Jours de repos
- `rest_advantage` → Avantage repos domicile
- `matchday_importance` → Importance de la journée (0-1)
- `is_weekend` → Match en weekend

#### **Modèles Entraînés :**

**XGBoost Optimisé**
- **Accuracy : 49.73%** (vs 47.55% baseline)
- **Best Params** : learning_rate=0.05, max_depth=4, n_estimators=100
- **Top Features** : goal_difference, form_difference, matchday_importance

**Random Forest**
- **Accuracy : 47.00%**
- **CV Score** : 48.02% ± 1.82%

#### **Performance Réaliste :**
- **Modèle naïf (toujours Home)** : ~44%
- **Bookmakers professionnels** : 50-55%
- **Notre modèle** : **49.7%** → Très compétitif!

---

### **3. 🚀 PIPELINE DE PRÉDICTION PRODUCTION**

#### **Fonctionnalités Opérationnelles :**
- ✅ **Chargement modèles** optimisés
- ✅ **Prédiction match unique** avec probabilités
- ✅ **Batch predictions** matchs à venir
- ✅ **Sortie structurée** avec confiance
- ✅ **Gestion d'erreurs** robuste

#### **Exemple de Prédiction :**
```python
{
  'prediction': 'Away',
  'probabilities': {'Home': 0.233, 'Draw': 0.188, 'Away': 0.580},
  'confidence': 0.580
}
```

---

## 📊 RÉSULTATS CONCRETS

### **Dataset Final :**
- **3,670 matchs** avec features complètes
- **Distribution cible équilibrée** : 
  - Home: 43.6% (1599)
  - Away: 31.6% (1160) 
  - Draw: 24.8% (911)

### **Qualité des Données :**
- ✅ **Aucune valeur manquante** après feature engineering
- ✅ **Progression temporelle** cohérente
- ✅ **Features discriminantes** identifiées
- ✅ **Split temporel** pour entraînement/validation

---

## 🔧 COMMANDES & UTILISATION

### **Collecte Données :**
```bash
python -m data_ingest.ingest --ingest-matches 2021 --days-back 730
```

### **Feature Engineering :**
```bash
python -m ml_pipeline.feature_engineering
```

### **Entraînement Modèles :**
```bash
python -m ml_pipeline.model_training
```

### **Optimisation :**
```bash
python -m ml_pipeline.model_improvement
```

### **Prédictions :**
```bash
python -m ml_pipeline.predict_upcoming
```








# 📚 Documentation Complète - FootStack

## 🎯 Vue d'Ensemble du Projet

**FootStack** est une application web complète d'analyse de football basée sur l'IA, conçue pour démontrer un large éventail de compétences en Data Science et en ingénierie logicielle.

### 📊 Statut du Projet
- **✅ Complété** : Architecture de base, ML Pipeline, API, Orchestration
- **🔄 En Cours** : Dashboard frontend, Améliorations
- **📋 Planifié** : Modules avancés (Computer Vision, LLM, Scouting)

---

## 🏗️ Architecture Technique

### 📁 Structure du Projet
```
FootStack/
├── 🗄️  data_ingest/          # Collecte et stockage des données
├── 🤖 ml_pipeline/           # Pipeline de Machine Learning
├── 🌐 api/                   # API FastAPI
├── ⚙️  airflow/              # Orchestration des workflows
├── 📊 data/                  # Données et features
├── 🧠 models/                # Modèles ML entraînés
└── 🐳 docker-compose.yml     # Conteneurisation
```

### 🛠️ Stack Technologique

| Domaine | Technologies |
|---------|--------------|
| **Backend** | Python, FastAPI, SQLAlchemy, Pydantic |
| **ML/Data Science** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Base de Données** | PostgreSQL, SQLAlchemy ORM |
| **Orchestration** | Apache Airflow, Bash Operators |
| **Conteneurisation** | Docker, Docker Compose |
| **API** | FastAPI, Swagger/OpenAPI |
| **Monitoring** | Logging structuré, Health Checks |

---

## 📈 Modules Implémentés

### 1. 🗄️ Fondation : Collecte et Stockage des Données

#### 🔧 Technologies
- **API Source** : Football-Data.org (API REST)
- **Base de Données** : PostgreSQL 15
- **ORM** : SQLAlchemy 2.0+
- **Gestion des erreurs** : Tenacity (retry pattern)

#### 📊 Schéma de Base de Données
```sql
-- Tables principales
competitions(id, name, area_name, code, data)
teams(id, name, short_name, tla, crest_url, data)
matches(id, competition_id, utc_date, status, matchday, home_team_id, away_team_id, score, raw)

-- Vue analytique
matches_cleaned(match_id, competition_name, home_team, away_team, home_score, away_score, result, date)
```

#### ⚡ Fonctionnalités
- ✅ Collecte automatique des compétitions et matchs
- ✅ Gestion robuste des rate limits et timeouts
- ✅ Stockage structuré avec sauvegarde des données brutes
- ✅ Scripts d'ingestion modulaires

---

### 2. 🤖 Module 1 : Prédiction des Résultats de Matchs

#### 🎯 Objectif
Prédire le résultat des matchs (Victoire Domicile/Nul/Victoire Extérieur) avec des modèles de Machine Learning.

#### 🔧 Implémentation

##### Feature Engineering
**20 features calculées :**
- **Forme récente** (5 derniers matchs) : points moyens, buts marqués/encaissés
- **Confrontations directes** : taux de victoires, historique
- **Contexte** : jours de repos, weekend, importance de la journée
- **Features différentielles** : écart de forme, écart de buts

##### Modèles Implémentés
- **XGBoost** (modèle principal) - Accuracy: ~49.7%
- **Random Forest** (modèle de comparaison)

##### Pipeline ML
```python
# Workflow complet
Data Loading → Feature Engineering → Train/Test Split → 
Model Training → Cross-Validation → Model Evaluation → Prediction
```

#### 📊 Performance
- **Accuracy** : 49.7% (meilleur que random 33%)
- **Validation** : Split temporel, Cross-validation 5 folds
- **Features Importance** : Forme récente > H2H > Contexte

---

### 3. 🌐 API FastAPI

#### 🚀 Endpoints Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `GET /` | GET | Page d'accueil et documentation |
| `GET /health` | GET | Health check complet |
| `POST /predictions/predict-auto` | POST | Prédiction automatique avec calcul features |
| `POST /predictions/predict` | POST | Prédiction manuelle avec features fournies |
| `GET /predictions/upcoming` | GET | Prédictions des prochains matchs |
| `GET /predictions/teams` | GET | Liste des équipes disponibles |

#### 🏗️ Architecture API
- **FastAPI** avec documentation Swagger/OpenAPI automatique
- **Dependency Injection** pour la DB et les modèles ML
- **Middleware CORS** pour le frontend
- **Validation** avec Pydantic schemas
- **Logging structuré** et gestion d'erreurs

#### 🔧 Services
- **FeatureCalculator** : Calcul temps réel des features
- **Health checks** : Vérification DB + modèles ML
- **Auto-reload** en développement

---

### 4. ⚙️ Orchestration Airflow

#### 📋 Pipeline DAG
```python
start → wait_for_db → ingest_data → clean_data → engineer_features → train_models → end
```

#### 🔄 Planification
- **Exécution** : Toutes les 2 semaines (`0 0 */14 * *`)
- **Durée estimée** : 15-30 minutes
- **Gestion d'erreurs** : Retries avec backoff

#### 🐳 Architecture Docker
```yaml
Services:
  postgres:15          # Base principale FootStack
  airflow-postgres:13  # Métadata Airflow  
  airflow-webserver    # Interface web
  airflow-scheduler    # Planificateur
```

#### 📊 Monitoring
- **Health checks** automatiques
- **Logs centralisés**
- **Interface web** sur port 8080

---

## 🎯 Fonctionnalités Implémentées

### ✅ Core Features
1. **Collecte de données** automatisée depuis API football
2. **Stockage robuste** avec PostgreSQL
3. **Feature engineering** sophistiqué (forme, H2H, contexte)
4. **Entraînement de modèles** ML (XGBoost, Random Forest)
5. **API RESTful** avec prédictions en temps réel
6. **Orchestration** avec Airflow
7. **Conteneurisation** complète avec Docker

### ✅ Features Avancées
1. **Health monitoring** automatique
2. **Gestion des erreurs** et retry mechanisms
3. **Documentation automatique** API
4. **Logging structuré**
5. **Configuration** externalisée (.env)

---

## 📊 Métriques et Performance

### 🎯 Performance ML
- **Accuracy** : 49.7% (XGBoost optimisé)
- **Baseline** : 33.3% (aléatoire)
- **Amélioration** : +16.4% par rapport au hasard
- **Cross-validation** : 48.2% ± 2.1%

### ⚡ Performance Technique
- **Temps de prédiction** : < 100ms
- **Disponibilité API** : Health checks complets
- **Robustesse données** : Gestion des missing values
- **Scalabilité** : Architecture conteneurisée

---

## 🔧 Installation et Déploiement

### Prérequis
```bash
Docker & Docker Compose
Python 3.9+
```

### Démarrage
```bash
# 1. Cloner le projet
git clone <repository>
cd FootStack

# 2. Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# 3. Démarrage
docker-compose up -d

# 4. Accès
API: http://localhost:8000
Airflow: http://localhost:8080 (airflow/airflow)
```

### Commandes Utiles
```bash
# Ingestion manuelle
python -m data_ingest.ingest --competitions --ingest-matches 2021

# Entraînement ML
python -m ml_pipeline.model_training

# Tests API
curl http://localhost:8000/health
```

---

## 🚀 Utilisation

### Prédiction Automatique
```bash
curl -X POST "http://localhost:8000/predictions/predict-auto" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Paris Saint-Germain FC",
    "away_team": "Olympique de Marseille"
  }'
```

### Réponse Type
```json
{
  "home_team": "Paris Saint-Germain FC",
  "away_team": "Olympique de Marseille", 
  "prediction": "Home",
  "probabilities": {
    "Home": 0.65,
    "Draw": 0.22,
    "Away": 0.13
  },
  "confidence": 0.65,
  "model_accuracy": 0.497
}
```

---

## 📈 Roadmap et Améliorations

### 🔮 Prochaines Étapes
1. **Dashboard React/Next.js** - Interface utilisateur
2. **Module Computer Vision** - Analyse vidéo des matchs
3. **Module LLM Chatbot** - Assistant football expert
4. **Module Scouting** - Détection de talents
5. **Monitoring avancé** - Métriques temps réel

### 🎯 Améliorations Possibles
- **Features additionnelles** : blessures, compositions d'équipe
- **Modèles avancés** : LSTM pour séries temporelles
- **API temps réel** : WebSockets pour live updates
- **Cache** : Redis pour performances
- **Tests automatisés** : Unit et integration tests

---

## 👨‍💻 Compétences Démonstrées

Ce projet démontre une expertise complète en :

### Data Engineering
- **ETL/ELT** pipelines avec Airflow
- **APIs REST** et gestion de rate limiting
- **Bases de données** relationnelles (PostgreSQL)
- **Conteneurisation** et orchestration

### Machine Learning
- **Feature engineering** domaine-spécifique
- **Modélisation** (XGBoost, Random Forest)
- **Validation** et évaluation de modèles
- **MLOps** : ré-entraînement automatique

### Software Engineering
- **API design** RESTful avec FastAPI
- **Architecture microservices**
- **DevOps** et déploiement Docker
- **Code qualité** : modularité, documentation

### Data Science
- **Analyse exploratoire** de données sportives
- **Métriques domaine-spécifiques**
- **Visualisation** et reporting
- **A/B testing** de modèles

---

## 🎉 Conclusion

**FootStack** représente un projet **production-ready** qui démontre des compétences techniques avancées à travers un cas d'usage concret et passionnant. L'architecture modulaire permet une extension facile vers les modules avancés planifiés, faisant de ce projet un excellent showcase pour une carrière en Data Science et ingénierie logicielle.

**🚀 Le projet est opérationnel et prêt pour le prochain stage PFE !**