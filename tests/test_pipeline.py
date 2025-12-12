import pytest
import numpy as np
import tempfile
import os
from core.logistic_regression import LogisticRegression
from core.decision_tree import DecisionTree
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from utils.processing import DataProcessor


@pytest.fixture
def sample_data():
    """Generate sample binary classification data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


class TestTrainer:
    """Test suite for Trainer class"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        model = LogisticRegression()
        trainer = Trainer(model)
        
        assert trainer.model == model
        assert trainer.preprocessor is not None
        assert isinstance(trainer.preprocessor, DataProcessor)
    
    def test_initialization_with_preprocessor(self):
        """Test trainer initialization with custom preprocessor"""
        model = LogisticRegression()
        preprocessor = DataProcessor()
        trainer = Trainer(model, preprocessor)
        
        assert trainer.preprocessor == preprocessor
    
    def test_train_with_normalization(self, sample_data):
        """Test training with normalization"""
        X_train, X_test, y_train, y_test = sample_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        trainer = Trainer(model)
        
        trained_model = trainer.train(X_train, y_train, normalize=True)
        
        assert trained_model.is_trained
        assert trainer.preprocessor.scaler_mean is not None
        assert trainer.preprocessor.scaler_std is not None
    
    def test_train_without_normalization(self, sample_data):
        """Test training without normalization"""
        X_train, X_test, y_train, y_test = sample_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        trainer = Trainer(model)
        
        trained_model = trainer.train(X_train, y_train, normalize=False)
        
        assert trained_model.is_trained
        assert trainer.preprocessor.scaler_mean is None
    
    def test_save_and_load_model(self, sample_data):
        """Test model saving and loading"""
        X_train, X_test, y_train, y_test = sample_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        trainer = Trainer(model)
        
        # Train model
        trainer.train(X_train, y_train, normalize=True)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            trainer.save_model(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Load model
            new_trainer = Trainer(LogisticRegression())
            new_trainer.load_model(tmp_path)
            
            assert new_trainer.model.is_trained
            assert new_trainer.preprocessor.scaler_mean is not None
            
            # Test predictions are the same
            original_pred = trainer.model.predict(X_test)
            loaded_pred = new_trainer.model.predict(X_test)
            assert np.array_equal(original_pred, loaded_pred)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_train_decision_tree(self, sample_data):
        """Test training with decision tree"""
        X_train, X_test, y_train, y_test = sample_data
        model = DecisionTree(max_depth=5)
        trainer = Trainer(model)
        
        trained_model = trainer.train(X_train, y_train, normalize=False)
        
        assert trained_model.is_trained
        assert trained_model.root is not None


class TestEvaluator:
    """Test suite for Evaluator class"""
    
    def test_initialization(self):
        """Test evaluator initialization"""
        model = LogisticRegression()
        evaluator = Evaluator(model)
        
        assert evaluator.model == model
        assert evaluator.preprocessor is not None
        assert evaluator.metrics is not None
    
    def test_initialization_with_preprocessor(self):
        """Test evaluator initialization with custom preprocessor"""
        model = LogisticRegression()
        preprocessor = DataProcessor()
        evaluator = Evaluator(model, preprocessor)
        
        assert evaluator.preprocessor == preprocessor
    
    def test_evaluate_with_normalization(self, sample_data):
        """Test evaluation with normalization"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        trainer = Trainer(model)
        trained_model = trainer.train(X_train, y_train, normalize=True)
        
        # Evaluate
        evaluator = Evaluator(trained_model, trainer.preprocessor)
        results = evaluator.evaluate(X_test, y_test, normalize=True)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        
        for value in results.values():
            assert 0 <= value <= 1
    
    def test_evaluate_without_normalization(self, sample_data):
        """Test evaluation without normalization"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        trainer = Trainer(model)
        trained_model = trainer.train(X_train, y_train, normalize=False)
        
        # Evaluate
        evaluator = Evaluator(trained_model)
        results = evaluator.evaluate(X_test, y_test, normalize=False)
        
        assert 'accuracy' in results
        assert all(0 <= value <= 1 for value in results.values())
    
    def test_print_evaluation(self, sample_data, capsys):
        """Test evaluation printing"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        trainer = Trainer(model)
        trained_model = trainer.train(X_train, y_train, normalize=True)
        
        # Evaluate and print
        evaluator = Evaluator(trained_model, trainer.preprocessor)
        results = evaluator.evaluate(X_test, y_test, normalize=True)
        evaluator.print_evaluation(results)
        
        # Check output
        captured = capsys.readouterr()
        assert 'RÉSULTATS D\'ÉVALUATION' in captured.out
        assert 'Accuracy' in captured.out
        assert 'Precision' in captured.out
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Combine train and test for CV
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        evaluator = Evaluator(model)
        
        avg_scores = evaluator.cross_validate(X, y, k_folds=3)
        
        assert 'accuracy' in avg_scores
        assert 'precision' in avg_scores
        assert 'recall' in avg_scores
        assert 'f1_score' in avg_scores
        
        for value in avg_scores.values():
            assert 0 <= value <= 1
    
    def test_evaluate_decision_tree(self, sample_data):
        """Test evaluation with decision tree"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model
        model = DecisionTree(max_depth=5)
        trainer = Trainer(model)
        trained_model = trainer.train(X_train, y_train, normalize=False)
        
        # Evaluate
        evaluator = Evaluator(trained_model, trainer.preprocessor)
        results = evaluator.evaluate(X_test, y_test, normalize=False)
        
        assert all(0 <= value <= 1 for value in results.values())
