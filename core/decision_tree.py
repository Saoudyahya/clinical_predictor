from core.model import BaseModel
import numpy as np
from typing import Optional


class Node:
    """Nœud de l'arbre de décision"""

    def __init__(self, feature_index: Optional[int] = None, threshold: Optional[float] = None,
                 left=None, right=None, value: Optional[float] = None):
        self.feature_index = feature_index  # Index de la feature pour la division
        self.threshold = threshold  # Seuil de division
        self.left = left  # Sous-arbre gauche
        self.right = right  # Sous-arbre droit
        self.value = value  # Valeur de prédiction (pour les feuilles)


class DecisionTree(BaseModel):
    """Implémentation d'un arbre de décision pour la classification binaire"""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'gini'):
        """
        Args:
            max_depth: Profondeur maximale de l'arbre
            min_samples_split: Nombre minimum d'échantillons pour diviser un nœud
            min_samples_leaf: Nombre minimum d'échantillons dans une feuille
            criterion: Critère de division ('gini' ou 'entropy')
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None

    def _gini(self, y: np.ndarray) -> float:
        """Calcule l'impureté de Gini"""
        if len(y) == 0:
            return 0

        p1 = np.sum(y == 1) / len(y)
        p0 = 1 - p1
        return 1 - (p0 ** 2 + p1 ** 2)

    def _entropy(self, y: np.ndarray) -> float:
        """Calcule l'entropie"""
        if len(y) == 0:
            return 0

        p1 = np.sum(y == 1) / len(y)
        p0 = 1 - p1

        entropy = 0
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)

        return entropy

    def _impurity(self, y: np.ndarray) -> float:
        """Calcule l'impureté selon le critère choisi"""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Critère inconnu: {self.criterion}")

    def _information_gain(self, y: np.ndarray, left_mask: np.ndarray,
                          right_mask: np.ndarray) -> float:
        """Calcule le gain d'information d'une division"""
        parent_impurity = self._impurity(y)

        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            return 0

        left_impurity = self._impurity(y[left_mask])
        right_impurity = self._impurity(y[right_mask])

        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

        return parent_impurity - weighted_impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Trouve la meilleure division"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                # Vérifier les contraintes min_samples_leaf
                if (np.sum(left_mask) < self.min_samples_leaf or
                        np.sum(right_mask) < self.min_samples_leaf):
                    continue

                gain = self._information_gain(y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Construit l'arbre de décision récursivement"""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Conditions d'arrêt
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Trouver la meilleure division
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # Si aucune division n'améliore l'impureté, créer une feuille
        if best_gain == 0 or best_feature is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Diviser les données
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        # Construire les sous-arbres
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entraîne l'arbre de décision"""
        self.root = self._build_tree(X, y)
        self.is_trained = True

    def _predict_sample(self, x: np.ndarray, node: Node) -> float:
        """Prédit pour un seul échantillon"""
        # Si c'est une feuille, retourner la valeur
        if node.value is not None:
            return node.value

        # Sinon, descendre dans l'arbre
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités de prédiction"""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné")

        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions binaires"""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)

    def get_tree_depth(self) -> int:
        """Retourne la profondeur de l'arbre"""

        def _depth(node: Node) -> int:
            if node is None or node.value is not None:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def count_leaves(self) -> int:
        """Compte le nombre de feuilles dans l'arbre"""

        def _count(node: Node) -> int:
            if node is None:
                return 0
            if node.value is not None:
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root)