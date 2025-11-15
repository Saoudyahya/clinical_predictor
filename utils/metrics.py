import numpy as np
from typing import Dict


class Metrics:
    """Classe pour calculer les métriques de performance"""

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule l'accuracy"""
        return np.mean(y_true == y_pred)

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcule la matrice de confusion"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return np.array([[tn, fp], [fn, tp]])

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la précision"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule le recall (sensibilité)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule le F1-score"""
        prec = Metrics.precision(y_true, y_pred)
        rec = Metrics.recall(y_true, y_pred)

        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    @staticmethod
    def get_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Retourne toutes les métriques"""
        return {
            'accuracy': Metrics.accuracy(y_true, y_pred),
            'precision': Metrics.precision(y_true, y_pred),
            'recall': Metrics.recall(y_true, y_pred),
            'f1_score': Metrics.f1_score(y_true, y_pred)
        }
