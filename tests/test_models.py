import pytest
import numpy as np
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.decision_tree import DecisionTree
import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

@pytest.fixture
def sample_data():
    """Generate sample binary classification data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y


@pytest.fixture
def train_test_data(sample_data):
    """Split data into train and test sets"""
    X, y = sample_data
    split_idx = int(0.8 * len(X))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


class TestLogisticRegression:
    """Test suite for Logistic Regression"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = LogisticRegression(learning_rate=0.01, n_iterations=100)
        assert model.learning_rate == 0.01
        assert model.n_iterations == 100
        assert not model.is_trained
    
    def test_sigmoid(self):
        """Test sigmoid function"""
        model = LogisticRegression()
        z = np.array([0, 1, -1, 100, -100])
        result = model.sigmoid(z)
        
        assert np.allclose(result[0], 0.5)
        assert result[1] > 0.5
        assert result[2] < 0.5
        assert np.allclose(result[3], 1.0)
        assert np.allclose(result[4], 0.0)
    
    def test_fit(self, train_test_data):
        """Test model training"""
        X_train, X_test, y_train, y_test = train_test_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        
        model.fit(X_train, y_train)
        
        assert model.is_trained
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.weights) == X_train.shape[1]
    
    def test_predict(self, train_test_data):
        """Test predictions"""
        X_train, X_test, y_train, y_test = train_test_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, train_test_data):
        """Test probability predictions"""
        X_train, X_test, y_train, y_test = train_test_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        
        assert len(probas) == len(X_test)
        assert all(0 <= p <= 1 for p in probas)
    
    def test_predict_without_training(self, sample_data):
        """Test that prediction fails without training"""
        X, y = sample_data
        model = LogisticRegression()
        
        with pytest.raises(ValueError, match="Le modèle n'est pas entraîné"):
            model.predict(X)


class TestNeuralNetwork:
    """Test suite for Neural Network"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = NeuralNetwork(hidden_layers=[32, 16], learning_rate=0.01)
        assert model.hidden_layers == [32, 16]
        assert model.learning_rate == 0.01
        assert not model.is_trained
    
    def test_sigmoid(self):
        """Test sigmoid function"""
        model = NeuralNetwork()
        z = np.array([0, 1, -1])
        result = model.sigmoid(z)
        
        assert np.allclose(result[0], 0.5)
        assert result[1] > 0.5
        assert result[2] < 0.5
    
    def test_fit(self, train_test_data):
        """Test model training"""
        X_train, X_test, y_train, y_test = train_test_data
        model = NeuralNetwork(hidden_layers=[16], learning_rate=0.1, n_iterations=100)
        
        model.fit(X_train, y_train)
        
        assert model.is_trained
        assert len(model.weights) > 0
        assert len(model.biases) > 0
    
    def test_predict(self, train_test_data):
        """Test predictions"""
        X_train, X_test, y_train, y_test = train_test_data
        model = NeuralNetwork(hidden_layers=[16], learning_rate=0.1, n_iterations=150)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, train_test_data):
        """Test probability predictions"""
        X_train, X_test, y_train, y_test = train_test_data
        model = NeuralNetwork(hidden_layers=[16], learning_rate=0.1, n_iterations=150)
        
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        
        assert len(probas) == len(X_test)
        assert all(0 <= p <= 1 for p in probas)


class TestDecisionTree:
    """Test suite for Decision Tree"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = DecisionTree(max_depth=5, min_samples_split=2)
        assert model.max_depth == 5
        assert model.min_samples_split == 2
        assert not model.is_trained
    
    def test_gini(self):
        """Test Gini impurity calculation"""
        model = DecisionTree()
        
        # Pure node (all 1s)
        y_pure = np.array([1, 1, 1, 1])
        assert np.allclose(model._gini(y_pure), 0.0)
        
        # Mixed node (50-50)
        y_mixed = np.array([0, 0, 1, 1])
        assert np.allclose(model._gini(y_mixed), 0.5)
    
    def test_entropy(self):
        """Test entropy calculation"""
        model = DecisionTree(criterion='entropy')
        
        # Pure node
        y_pure = np.array([1, 1, 1, 1])
        assert np.allclose(model._entropy(y_pure), 0.0)
        
        # Mixed node (50-50)
        y_mixed = np.array([0, 0, 1, 1])
        assert model._entropy(y_mixed) > 0
    
    def test_fit(self, train_test_data):
        """Test model training"""
        X_train, X_test, y_train, y_test = train_test_data
        model = DecisionTree(max_depth=5)
        
        model.fit(X_train, y_train)
        
        assert model.is_trained
        assert model.root is not None
    
    def test_predict(self, train_test_data):
        """Test predictions"""
        X_train, X_test, y_train, y_test = train_test_data
        model = DecisionTree(max_depth=5)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_tree_depth(self, train_test_data):
        """Test tree depth calculation"""
        X_train, X_test, y_train, y_test = train_test_data
        model = DecisionTree(max_depth=3)
        
        model.fit(X_train, y_train)
        depth = model.get_tree_depth()
        
        assert depth <= 3
        assert depth >= 0
    
    def test_count_leaves(self, train_test_data):
        """Test leaf counting"""
        X_train, X_test, y_train, y_test = train_test_data
        model = DecisionTree(max_depth=3)
        
        model.fit(X_train, y_train)
        leaves = model.count_leaves()
        
        assert leaves > 0
