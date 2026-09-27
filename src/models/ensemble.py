import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

class AudioAuthenticationEnsemble:
    def __init__(self):
        # We use XGBoost because it naturally handles diverse feature scales
        # and outputs feature importance, aiding explainability.
        self.weighter = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric='logloss'
        )
        self.feature_names = []

    def train(self, X, y, feature_names=None):
        """
        Train the ensemble weighter.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) binary labels (0=real, 1=synthetic)
        """
        print(f"Training ensemble on {X.shape[0]} samples with {X.shape[1]} features...")
        self.weighter.fit(X, y)
        if feature_names:
            self.feature_names = feature_names
        print("Training complete.")

    def evaluate(self, X_val, y_val):
        """
        Evaluate the ensemble.
        """
        preds_proba = self.predict_proba(X_val)
        auc = roc_auc_score(y_val, preds_proba)
        loss = log_loss(y_val, preds_proba)
        print(f"Validation AUC: {auc:.4f} | Log Loss: {loss:.4f}")
        return auc

    def predict_proba(self, X):
        """
        Predict the probability of being synthetic (cm-score).
        """
        # Return probability of class 1 (synthetic)
        return self.weighter.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """
        Return the importance of each tool/feature in the ensemble.
        """
        importance = self.weighter.feature_importances_
        if self.feature_names:
            return sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True)
        return importance
