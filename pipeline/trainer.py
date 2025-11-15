from core.model import BaseModel
from utils.processing import DataProcessor
import numpy as np


class Trainer:
    """Classe pour entraîner les modèles"""

    def __init__(self, model: BaseModel, preprocessor: DataProcessor = None):
        self.model = model
        self.preprocessor = preprocessor or DataProcessor()
        self.training_history = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              normalize: bool = True) -> BaseModel:
        """Entraîne le modèle"""

        # Prétraitement
        if normalize:
            X_train = self.preprocessor.normalize(X_train, fit=True)

        # Entraînement
        print("Début de l'entraînement...")
        self.model.fit(X_train, y_train)
        print("Entraînement terminé!")

        return self.model

    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor
            }, f)
        print(f"Modèle sauvegardé dans {filepath}")

    def load_model(self, filepath: str):
        """Charge un modèle"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.preprocessor = data['preprocessor']
        print(f"Modèle chargé depuis {filepath}")