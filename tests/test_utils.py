import pytest
import numpy as np
import pandas as pd
from utils.metrics import Metrics
from utils.processing import DataProcessor


class TestMetrics:
    """Test suite for Metrics class"""
    
    @pytest.fixture
    def perfect_predictions(self):
        """Perfect predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        return y_true, y_pred
    
    @pytest.fixture
    def imperfect_predictions(self):
        """Imperfect predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0])
        return y_true, y_pred
    
    def test_accuracy_perfect(self, perfect_predictions):
        """Test accuracy with perfect predictions"""
        y_true, y_pred = perfect_predictions
        accuracy = Metrics.accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_accuracy_imperfect(self, imperfect_predictions):
        """Test accuracy with imperfect predictions"""
        y_true, y_pred = imperfect_predictions
        accuracy = Metrics.accuracy(y_true, y_pred)
        assert 0 <= accuracy <= 1
        assert accuracy == 0.625  # 5 correct out of 8
    
    def test_confusion_matrix(self, imperfect_predictions):
        """Test confusion matrix calculation"""
        y_true, y_pred = imperfect_predictions
        cm = Metrics.confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert np.all(cm >= 0)
        assert cm.sum() == len(y_true)
    
    def test_precision(self, imperfect_predictions):
        """Test precision calculation"""
        y_true, y_pred = imperfect_predictions
        precision = Metrics.precision(y_true, y_pred)
        
        assert 0 <= precision <= 1
    
    def test_recall(self, imperfect_predictions):
        """Test recall calculation"""
        y_true, y_pred = imperfect_predictions
        recall = Metrics.recall(y_true, y_pred)
        
        assert 0 <= recall <= 1
    
    def test_f1_score(self, imperfect_predictions):
        """Test F1-score calculation"""
        y_true, y_pred = imperfect_predictions
        f1 = Metrics.f1_score(y_true, y_pred)
        
        assert 0 <= f1 <= 1
    
    def test_get_all_metrics(self, imperfect_predictions):
        """Test getting all metrics at once"""
        y_true, y_pred = imperfect_predictions
        metrics = Metrics.get_all_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        for value in metrics.values():
            assert 0 <= value <= 1
    
    def test_edge_case_all_zeros(self):
        """Test edge case with all zeros"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        accuracy = Metrics.accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_edge_case_no_positive_predictions(self):
        """Test edge case with no positive predictions"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        
        assert precision == 0.0
        assert recall == 0.0


class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        return np.random.randn(100, 5) * 10 + 50
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = DataProcessor()
        assert processor.scaler_mean is None
        assert processor.scaler_std is None
    
    def test_normalize_fit(self, sample_data):
        """Test normalization with fitting"""
        processor = DataProcessor()
        X_normalized = processor.normalize(sample_data, fit=True)
        
        assert processor.scaler_mean is not None
        assert processor.scaler_std is not None
        assert X_normalized.shape == sample_data.shape
        
        # Check that mean is close to 0 and std is close to 1
        assert np.allclose(np.mean(X_normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_normalized, axis=0), 1, atol=1e-10)
    
    def test_normalize_transform(self, sample_data):
        """Test normalization without fitting"""
        processor = DataProcessor()
        
        # First fit
        X_train = sample_data[:80]
        processor.normalize(X_train, fit=True)
        
        # Then transform test data
        X_test = sample_data[80:]
        X_test_normalized = processor.normalize(X_test, fit=False)
        
        assert X_test_normalized.shape == X_test.shape
    
    def test_normalize_zero_std(self):
        """Test normalization with zero standard deviation"""
        processor = DataProcessor()
        X = np.array([[1, 2], [1, 3], [1, 4]])  # First column has zero std
        
        X_normalized = processor.normalize(X, fit=True)
        
        # Should not produce NaN values
        assert not np.any(np.isnan(X_normalized))
    
    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy"""
        processor = DataProcessor()
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })
        
        result = processor.handle_missing_values(data, strategy='mean')
        
        assert not result.isnull().any().any()
        assert result['A'].iloc[2] == pytest.approx(2.333, rel=0.01)
    
    def test_handle_missing_values_median(self):
        """Test handling missing values with median strategy"""
        processor = DataProcessor()
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })
        
        result = processor.handle_missing_values(data, strategy='median')
        
        assert not result.isnull().any().any()
    
    def test_handle_missing_values_drop(self):
        """Test handling missing values with drop strategy"""
        processor = DataProcessor()
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, 6, 7, 8]
        })
        
        result = processor.handle_missing_values(data, strategy='drop')
        
        assert len(result) == 3
        assert not result.isnull().any().any()
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test handling missing values with invalid strategy"""
        processor = DataProcessor()
        data = pd.DataFrame({'A': [1, 2, np.nan]})
        
        with pytest.raises(ValueError, match="Stratégie inconnue"):
            processor.handle_missing_values(data, strategy='invalid')
    
    def test_remove_outliers(self, sample_data):
        """Test outlier removal"""
        processor = DataProcessor()
        
        # Add some outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers[0] = [1000, 1000, 1000, 1000, 1000]
        
        X_clean, mask = processor.remove_outliers(data_with_outliers, threshold=3.0)
        
        assert len(X_clean) < len(data_with_outliers)
        assert len(mask) == len(data_with_outliers)
        assert not mask[0]  # First row should be marked as outlier
    
    def test_remove_outliers_no_outliers(self, sample_data):
        """Test outlier removal with no outliers"""
        processor = DataProcessor()
        
        X_clean, mask = processor.remove_outliers(sample_data, threshold=10.0)
        
        # With high threshold, no outliers should be removed
        assert len(X_clean) == len(sample_data)
        assert np.all(mask)
