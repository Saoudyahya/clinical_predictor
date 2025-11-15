import pandas as pd
import numpy as np
from typing import Tuple, Optional


class ClinicalDataset:
    """Classe pour gérer les données cliniques"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self) -> pd.DataFrame:
        """Charge les données depuis un fichier CSV"""
        self.data = pd.read_csv(self.data_path)
        return self.data

    def split_features_target(self, target_column: str = 'target'):
        """Sépare les features et la cible"""
        if self.data is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")

        self.X = self.data.drop(columns=[target_column]).values
        self.y = self.data[target_column].values
        return self.X, self.y

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Divise les données en train/test"""
        from sklearn.model_selection import train_test_split

        if self.X is None or self.y is None:
            raise ValueError("Features et target non définis.")

        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)