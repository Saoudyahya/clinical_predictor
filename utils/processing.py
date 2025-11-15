import numpy as np
import pandas as pd
from typing import Tuple


class DataProcessor:
    """Classe pour le prétraitement des données"""

    def __init__(self):
        self.scaler_mean = None
        self.scaler_std = None

    def normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalise les données (z-score)"""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0)

        # Éviter la division par zéro
        std_safe = np.where(self.scaler_std == 0, 1, self.scaler_std)
        return (X - self.scaler_mean) / std_safe

    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Gère les valeurs manquantes"""
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")

    def remove_outliers(self, X: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Supprime les outliers avec la méthode z-score"""
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        mask = np.all(z_scores < threshold, axis=1)
        return X[mask], mask



