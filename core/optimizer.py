import numpy as np


class Optimizer:
    """Classe de base pour les optimiseurs"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate


class GradientDescent(Optimizer):
    """Descente de gradient standard"""

    def update(self, weights, bias, dw, db):
        """Met à jour les poids et biais"""
        weights = weights - self.learning_rate * dw
        bias = bias - self.learning_rate * db
        return weights, bias


class Adam(Optimizer):
    """Optimiseur Adam"""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0

    def update(self, weights, bias, dw, db):
        """Met à jour les poids avec Adam"""
        self.t += 1

        # Mise à jour des moments pour les poids
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)

        # Correction du biais
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)

        # Mise à jour des poids
        weights = weights - self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # Même chose pour le biais
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        bias = bias - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return weights, bias