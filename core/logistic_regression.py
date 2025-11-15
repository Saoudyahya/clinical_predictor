from core.model import BaseModel
from core.optimizer import GradientDescent
import numpy as np


class LogisticRegression(BaseModel):
    """Implémentation de la régression logistique"""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.optimizer = GradientDescent(learning_rate)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Fonction sigmoid"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entraîne le modèle de régression logistique"""
        n_samples, n_features = X.shape

        # Initialisation des poids
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Prédiction
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Calcul des gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Mise à jour des poids
            self.weights, self.bias = self.optimizer.update(
                self.weights, self.bias, dw, db
            )

        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités"""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné")

        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions binaires"""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)
