import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballModelTrainer:
    def __init__(self, features_file: str = "data/features_dataset.csv"):
        self.features_file = features_file
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Load features dataset"""
        logger.info(f"Loading features from {self.features_file}")
        df = pd.read_csv(self.features_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for training"""
        logger.info("Preparing features for training...")
        
        # Sélectionner les features numériques
        feature_columns = [
            'home_points_avg_5', 'home_goals_for_avg_5', 'home_goals_against_avg_5',
            'home_goal_diff_avg_5', 'home_recent_form', 'away_points_avg_5',
            'away_goals_for_avg_5', 'away_goals_against_avg_5', 'away_goal_diff_avg_5',
            'away_recent_form', 'form_difference', 'goal_difference',
            'h2h_win_rate_home', 'h2h_win_rate_away', 'h2h_matches_played',
            'home_days_rest', 'away_days_rest', 'rest_advantage', 
            'matchday_importance', 'is_weekend'
        ]
        
        # Encoder la variable cible
        y = self.label_encoder.fit_transform(df['target'])
        
        # Features
        X = df[feature_columns]
        
        # Split temporel (80% anciens, 20% récents)
        split_date = df['date'].quantile(0.8)
        train_mask = df['date'] <= split_date
        test_mask = df['date'] > split_date
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        logger.info(f"Training set: {len(X_train)} matches")
        logger.info(f"Test set: {len(X_test)} matches")
        logger.info(f"Feature shape: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
        
        results = {
            'model': rf_model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(X_train.columns, rf_model.feature_importances_)),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
        logger.info(f"Random Forest CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
        
        results = {
            'model': xgb_model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(X_train.columns, xgb_model.feature_importances_)),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        logger.info(f"XGBoost CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def evaluate_models(self, y_test, rf_results, xgb_results):
        """Compare model performance"""
        logger.info("\n" + "="*50)
        logger.info("MODEL COMPARISON")
        logger.info("="*50)
        
        print(f"\nRandom Forest Performance:")
        print(f"Test Accuracy: {rf_results['accuracy']:.4f}")
        print(f"Cross-Validation: {rf_results['cv_mean']:.4f} (+/- {rf_results['cv_std'] * 2:.4f})")
        
        print(f"\nXGBoost Performance:")
        print(f"Test Accuracy: {xgb_results['accuracy']:.4f}")
        print(f"Cross-Validation: {xgb_results['cv_mean']:.4f} (+/- {xgb_results['cv_std'] * 2:.4f})")
        
        # Feature importance
        print(f"\nTop 10 Random Forest Features:")
        rf_importances = sorted(rf_results['feature_importance'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in rf_importances:
            print(f"  {feature}: {importance:.4f}")
        
        print(f"\nTop 10 XGBoost Features:")
        xgb_importances = sorted(xgb_results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in xgb_importances:
            print(f"  {feature}: {importance:.4f}")
    
    def train_all_models(self):
        """Train all models and compare performance"""
        # Load and prepare data
        df = self.load_data()
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features(df)
        
        # Train models
        rf_results = self.train_random_forest(X_train, X_test, y_train, y_test)
        xgb_results = self.train_xgboost(X_train, X_test, y_train, y_test)
        
        # Store results
        self.models = {
            'random_forest': rf_results['model'],
            'xgboost': xgb_results['model']
        }
        
        self.results = {
            'random_forest': rf_results,
            'xgboost': xgb_results
        }
        
        # Evaluate and compare
        self.evaluate_models(y_test, rf_results, xgb_results)
        
        return self.models, self.results
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            filename = f"{model_dir}/{name}_{timestamp}.joblib"
            joblib.dump(model, filename)
            logger.info(f"Saved {name} model to {filename}")
        
        # Save label encoder
        encoder_file = f"{model_dir}/label_encoder_{timestamp}.joblib"
        joblib.dump(self.label_encoder, encoder_file)
        logger.info(f"Saved label encoder to {encoder_file}")

if __name__ == "__main__":
    trainer = FootballModelTrainer()
    models, results = trainer.train_all_models()
    trainer.save_models()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)