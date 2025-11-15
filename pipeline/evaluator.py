from core.model import BaseModel
from utils.metrics import Metrics
from utils.processing import DataProcessor
import numpy as np


class Evaluator:
    """Classe pour évaluer les modèles"""

    def __init__(self, model: BaseModel, preprocessor: DataProcessor = None):
        self.model = model
        self.preprocessor = preprocessor or DataProcessor()
        self.metrics = Metrics()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 normalize: bool = True) -> dict:
        """Évalue le modèle"""

        # Prétraitement
        if normalize:
            X_test = self.preprocessor.normalize(X_test, fit=False)

        # Prédictions
        y_pred = self.model.predict(X_test)

        # Calcul des métriques
        results = self.metrics.get_all_metrics(y_test, y_pred)

        return results

    def print_evaluation(self, results: dict):
        """Affiche les résultats d'évaluation"""
        print("\n" + "=" * 50)
        print("RÉSULTATS D'ÉVALUATION")
        print("=" * 50)
        for metric, value in results.items():
            print(f"{metric.capitalize():15s}: {value:.4f}")
        print("=" * 50 + "\n")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> dict:
        """Validation croisée k-fold"""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"Fold {fold}/{k_folds}...")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Entraînement sur ce fold
            self.model.fit(X_train_fold, y_train_fold)

            # Évaluation
            results = self.evaluate(X_val_fold, y_val_fold, normalize=False)

            for metric, value in results.items():
                scores[metric].append(value)

        # Moyennes
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        return avg_scores