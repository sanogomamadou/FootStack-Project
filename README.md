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

