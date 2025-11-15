from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Classe de base abstraite pour tous les modèles"""

    def __init__(self):
        self.is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entraîne le modèle"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités de prédiction"""
        pass