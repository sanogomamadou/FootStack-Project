# airflow/dags/footstack_pipeline.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

default_args = {
    'owner': 'footstack',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': days_ago(1),
}

with DAG(
    'footstack_pipeline',
    default_args=default_args,
    description='Pipeline FootStack simple et fiable',
    schedule_interval='0 0 */14 * *',
    catchup=False,
    tags=['footstack', 'ml'],
) as dag:

    start = DummyOperator(task_id='start')
    
    # 1. Attendre que la base de données soit prête
    wait_for_db = BashOperator(
        task_id='wait_for_database',
        bash_command="""
echo "  Attente de la base de données..."
until PGPASSWORD=postgres psql -h postgres -U postgres -d footstack -c "SELECT 1;" > /dev/null 2>&1; do
    sleep 5
    echo "En attente de PostgreSQL..."
done
echo "  Base de données prête!"
""",
    )
    
    # 2. Ingestion des données
    ingest_data = BashOperator(
        task_id='ingest_all_competitions',
        bash_command="""
cd /opt/footstack && \
python -c '
import os
os.environ[\"DATABASE_URL\"] = \"postgresql://postgres:postgres@postgres:5432/footstack\"

from data_ingest.ingest import init_db, ingest_matches_for_competition
from data_ingest.db import SessionLocal

print(\"  Début ingestion données...\")
init_db()
session = SessionLocal()

competitions = [2001, 2021, 2014, 2019, 2002, 2015]
for comp_id in competitions:
    print(f\"  Compétition {comp_id}...\")
    ingest_matches_for_competition(session, comp_id, days_back=730)

session.close()
print(\"  Ingestion terminée\")
'
""",
    )
    
    # 3. Nettoyage des données
    clean_data = BashOperator(
        task_id='clean_data',
        bash_command="""
echo "  Nettoyage des données..."
PGPASSWORD=postgres psql -h postgres -U postgres -d footstack << 'EOSQL'
DROP TABLE IF EXISTS matches_cleaned;

CREATE TABLE matches_cleaned AS
SELECT
    m.id AS match_id,
    m.competition_id,
    m.utc_date::timestamp AS date,
    (m.raw -> 'competition' ->> 'name') AS competition_name,
    (m.raw -> 'season' ->> 'startDate') AS season_start,
    (m.raw -> 'season' ->> 'endDate') AS season_end,
    (m.raw -> 'homeTeam' ->> 'name') AS home_team,
    (m.raw -> 'awayTeam' ->> 'name') AS away_team,
    (m.score -> 'fullTime' ->> 'home')::int AS home_score,
    (m.score -> 'fullTime' ->> 'away')::int AS away_score,
    CASE 
        WHEN (m.score ->> 'winner') = 'HOME_TEAM' THEN 'Home'
        WHEN (m.score ->> 'winner') = 'AWAY_TEAM' THEN 'Away'
        WHEN (m.score ->> 'winner') = 'DRAW' THEN 'Draw'
        ELSE NULL 
    END AS result,
    m.matchday,
    (m.raw -> 'area' ->> 'name') AS country,
    (m.raw -> 'stage') AS stage
FROM matches m
WHERE m.status = 'FINISHED'
AND m.score IS NOT NULL
AND (m.raw -> 'homeTeam' ->> 'name') IS NOT NULL
AND (m.raw -> 'awayTeam' ->> 'name') IS NOT NULL;

SELECT '  matches_cleaned: ' || COUNT(*) || ' matchs' FROM matches_cleaned;
EOSQL
""",
    )
    
    # 4. Feature Engineering
    engineer_features = BashOperator(
        task_id='engineer_features',
        bash_command="""
cd /opt/footstack && \
python -c '
import os
os.environ[\"DATABASE_URL\"] = \"postgresql://postgres:postgres@postgres:5432/footstack\"
from ml_pipeline.feature_engineering import FootballFeatureEngineer
print(\"  Calcul des features...\")
engineer = FootballFeatureEngineer(\"postgresql://postgres:postgres@postgres:5432/footstack\")
features_df = engineer.build_features()
features_df.to_csv(\"/opt/footstack/data/features_dataset.csv\", index=False)
print(f\"  Features: {features_df.shape}\")
'
""",
    )
    
    # 5. Entraînement des modèles
    train_models = BashOperator(
        task_id='train_models',
        bash_command="""
cd /opt/footstack && \
python -c '
from ml_pipeline.model_training import FootballModelTrainer
print(\"  Entraînement des modèles...\")
trainer = FootballModelTrainer(\"/opt/footstack/data/features_dataset.csv\")
models, results = trainer.train_all_models()
trainer.save_models(\"/opt/footstack/models\")
print(\"  Modèles entraînés\")
'
""",
    )

    # 6. OPTIMISATION DES MODÈLES (NOUVELLE TÂCHE)
    optimize_models = BashOperator(
        task_id='optimize_models',
        bash_command="""
cd /opt/footstack && \
python -c '
from ml_pipeline.model_improvement import ModelImprover
print(\"  Optimisation des modèles...\")
improver = ModelImprover(\"/opt/footstack/data/features_dataset.csv\")
best_model, accuracy = improver.optimize_xgboost()
print(f\"  Modèle optimisé: Accuracy {accuracy:.4f}\")

# Sauvegarde des modèles optimisés
import joblib
joblib.dump(best_model, \"/opt/footstack/models/xgboost_optimized.joblib\")
joblib.dump(improver.label_encoder, \"/opt/footstack/models/label_encoder_optimized.joblib\")
print(\"  Modèles optimisés sauvegardés\")
'
""",
    )
    
    end = DummyOperator(task_id='end')

    # Workflow (AVEC OPTIMISATION)
    start >> wait_for_db >> ingest_data >> clean_data >> engineer_features >> train_models >> optimize_models >> end