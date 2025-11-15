from core.model import BaseModel
import numpy as np


class NeuralNetwork(BaseModel):
    """Réseau de neurones simple"""

    def __init__(self, hidden_layers: list = [64, 32], learning_rate: float = 0.01,
                 n_iterations: int = 1000):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = []
        self.biases = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def initialize_parameters(self, n_features: int):
        """Initialise les poids et biais du réseau"""
        layer_dims = [n_features] + self.hidden_layers + [1]

        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.01
            b = np.zeros((1, layer_dims[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward_propagation(self, X: np.ndarray):
        """Propagation avant"""
        activations = [X]
        z_values = []

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        return activations, z_values

    def backward_propagation(self, X: np.ndarray, y: np.ndarray, activations, z_values):
        """Rétropropagation"""
        m = X.shape[0]
        y = y.reshape(-1, 1)

        gradients_w = []
        gradients_b = []

        # Gradient de la dernière couche
        dz = activations[-1] - y

        for i in reversed(range(len(self.weights))):
            dw = (1 / m) * np.dot(activations[i].T, dz)
            db = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)

            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.sigmoid_derivative(z_values[i - 1])

        return gradients_w, gradients_b

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entraîne le réseau de neurones"""
        self.initialize_parameters(X.shape[1])

        for iteration in range(self.n_iterations):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)

            # Backward propagation
            gradients_w, gradients_b = self.backward_propagation(X, y, activations, z_values)

            # Mise à jour des poids
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités"""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné")

        activations, _ = self.forward_propagation(X)
        return activations[-1].flatten()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions binaires"""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)