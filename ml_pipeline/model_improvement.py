import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

class ModelImprover:
    def __init__(self, features_file="data/features_dataset.csv"):
        self.features_file = features_file
        self.df = pd.read_csv(features_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.label_encoder = LabelEncoder()
        
    def prepare_features(self):
        feature_columns = [
            'home_points_avg_5', 'home_goals_for_avg_5', 'home_goals_against_avg_5',
            'home_goal_diff_avg_5', 'home_recent_form', 'away_points_avg_5',
            'away_goals_for_avg_5', 'away_goals_against_avg_5', 'away_goal_diff_avg_5',
            'away_recent_form', 'form_difference', 'goal_difference',
            'h2h_win_rate_home', 'h2h_win_rate_away', 'h2h_matches_played',
            'home_days_rest', 'away_days_rest', 'rest_advantage', 
            'matchday_importance', 'is_weekend'
        ]
        
        X = self.df[feature_columns]
        y = self.label_encoder.fit_transform(self.df['target'])  # ENCODEUR AJOUTÉ
        
        # Split temporel plus strict
        split_date = self.df['date'].quantile(0.85)  # 85% pour l'entraînement
        train_mask = self.df['date'] <= split_date
        test_mask = self.df['date'] > split_date
        
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask]
    
    def optimize_xgboost(self):
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Grid search simplifié pour éviter les erreurs
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
        
        return best_model, accuracy

if __name__ == "__main__":
    improver = ModelImprover()
    best_model, accuracy = improver.optimize_xgboost()
    
    # Sauvegarder le modèle optimisé et l'encodeur
    joblib.dump(best_model, "models/xgboost_optimized.joblib")
    joblib.dump(improver.label_encoder, "models/label_encoder_optimized.joblib")
    print("Optimized model and encoder saved!")