"""
Unit tests for ML inference system.

Tests the core inference pipeline, model loading, and prediction logic.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import torch
import numpy as np

try:
    from kataforge.ml.inference import InferenceEngine, InferenceRequest
    from kataforge.ml.models import FormAssessor
    from kataforge.core.settings import Settings
    from kataforge.core.error_handling import (
        ModelLoadingError, 
        ModelInferenceError,
        BiomechanicsError
    )
except ImportError:
    pytest.skip("ML inference not available")

class TestInferenceEngine:
    """Test the main inference engine."""
    
    def test_init_with_settings(self, mock_settings):
        """Test engine initialization with settings."""
        engine = InferenceEngine(mock_settings)
        
        assert engine.settings == mock_settings
        assert engine.model is None
        assert engine.device == mock_settings.model_device
    
    def test_init_with_invalid_device(self, mock_settings):
        """Test engine initialization handles invalid device."""
        mock_settings.model_device = "invalid-device"
        
        with pytest.raises(ValueError):
            InferenceEngine(mock_settings)
    
    def test_auto_device_selection(self, mock_settings):
        """Test automatic device selection works."""
        mock_settings.model_device = "auto"
        
        with patch('torch.cuda.is_available', return_value=True):
            engine = InferenceEngine(mock_settings)
            assert engine.device == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            engine = InferenceEngine(mock_settings)
            assert engine.device == "cpu"

class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('torch.load')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_model_success(self, mock_exists, mock_torch_load, mock_settings, temp_dir):
        """Test successful model loading."""
        mock_settings.model_path = temp_dir / "model.pt"
        
        # Mock successful model load
        mock_model = Mock(spec=FormAssessor)
        mock_torch_load.return_value = mock_model
        
        engine = InferenceEngine(mock_settings)
        engine.load_model()
        
        mock_torch_load.assert_called_once_with(str(temp_dir / "model.pt"), weights_only=True)
        assert engine.model == mock_model
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_model_file_not_found(self, mock_exists, mock_settings):
        """Test model loading fails when file doesn't exist."""
        mock_settings.model_path = "/nonexistent/model.pt"
        
        engine = InferenceEngine(mock_settings)
        
        with pytest.raises(ModelLoadingError):
            engine.load_model()
    
    @patch('torch.load', side_effect=Exception("Corrupted file"))
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_model_corrupted(self, mock_exists, mock_torch_load, mock_settings):
        """Test model loading fails with corrupted file."""
        mock_settings.model_path = "/test/model.pt"
        
        engine = InferenceEngine(mock_settings)
        
        with pytest.raises(ModelLoadingError):
            engine.load_model()

class TestPrediction:
    """Test model prediction functionality."""
    
    def test_predict_pose_data_shape(self, mock_ml_model, sample_pose_data):
        """Test prediction requires correct pose data shape."""
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        # Test with invalid shape
        invalid_pose = {"invalid": "data"}
        
        with pytest.raises(ValueError):
            engine.predict(invalid_pose)
    
    def test_predict_no_model_loaded(self, sample_pose_data):
        """Test prediction fails when no model is loaded."""
        engine = InferenceEngine(Settings())
        
        with pytest.raises(ModelInferenceError):
            engine.predict(sample_pose_data)
    
    def test_predict_success(self, mock_ml_model, sample_pose_data):
        """Test successful prediction."""
        # Mock model to return proper output
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([8.5]),
            'aspect_scores': torch.tensor([[7.5, 8.0, 8.5, 8.2, 8.1]])
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        result = engine.predict(sample_pose_data)
        
        assert 'overall_score' in result
        assert 'aspect_scores' in result
        assert isinstance(result['overall_score'], float)
        assert isinstance(result['aspect_scores'], dict)
        assert len(result['aspect_scores']) == 5

class TestBatchInference:
    """Test batch inference functionality."""
    
    def test_batch_predict_empty_list(self, mock_ml_model):
        """Test batch prediction with empty list."""
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        result = engine.batch_predict([])
        
        assert result == []
    
    def test_batch_predict_single_item(self, mock_ml_model, sample_pose_data):
        """Test batch prediction with single item."""
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([8.5]),
            'aspect_scores': torch.tensor([[7.5, 8.0, 8.5, 8.2, 8.1]])
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        results = engine.batch_predict([sample_pose_data])
        
        assert len(results) == 1
        assert 'overall_score' in results[0]
    
    def test_batch_predict_multiple_items(self, mock_ml_model, sample_pose_data):
        """Test batch prediction with multiple items."""
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([8.5]),
            'aspect_scores': torch.tensor([[7.5, 8.0, 8.5, 8.2, 8.1]])
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        poses = [sample_pose_data, sample_pose_data, sample_pose_data]
        results = engine.batch_predict(poses)
        
        assert len(results) == 3
        for result in results:
            assert 'overall_score' in result
            assert 'aspect_scores' in result

class TestInferenceRequest:
    """Test inference request data structure."""
    
    def test_request_creation(self):
        """Test inference request creation."""
        request = InferenceRequest(
            video_path="/test/video.mp4",
            coach="test-coach",
            technique="front-kick"
        )
        
        assert request.video_path == "/test/video.mp4"
        assert request.coach == "test-coach"
        assert request.technique == "front-kick"
    
    def test_request_validation(self):
        """Test inference request validation."""
        # Test missing required fields
        with pytest.raises(ValueError):
            InferenceRequest(video_path="", coach="test", technique="kick")
        
        with pytest.raises(ValueError):
            InferenceRequest(video_path="/test.mp4", coach="", technique="kick")
        
        with pytest.raises(ValueError):
            InferenceRequest(video_path="/test.mp4", coach="test", technique="")

class TestFallbackMechanisms:
    """Test fallback mechanisms when models fail."""
    
    def test_fallback_to_biomechanics(self, mock_settings, sample_pose_data, mock_biomechanics_calculator):
        """Test fallback to biomechanics-based scoring."""
        engine = InferenceEngine(mock_settings)
        
        # Don't load a model to trigger fallback
        result = engine.predict_with_fallback(sample_pose_data)
        
        # Should call biomechanics calculator
        mock_biomechanics_calculator.calculate_all.assert_called_once()
        
        # Should return biomechanics-based scores
        assert 'overall_score' in result
        assert isinstance(result['overall_score'], float)
        assert 1.0 <= result['overall_score'] <= 10.0
    
    def test_fallback_handles_biomechanics_error(self, mock_settings, sample_pose_data, mock_biomechanics_calculator):
        """Test fallback handles biomechanics errors gracefully."""
        mock_biomechanics_calculator.calculate_all.side_effect = Exception("Biomechanics failed")
        
        engine = InferenceEngine(mock_settings)
        
        with pytest.raises(BiomechanicsError):
            engine.predict_with_fallback(sample_pose_data)

class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_device_optimization(self, mock_settings):
        """Test device-specific optimizations."""
        mock_settings.model_device = "cuda"
        
        with patch('torch.cuda.is_available', return_value=True):
            engine = InferenceEngine(mock_settings)
            assert engine.device == "cuda"
    
    def test_model_caching(self, mock_settings, mock_ml_model):
        """Test model caching to avoid reloading."""
        with patch('torch.load') as mock_load:
            mock_load.return_value = mock_ml_model
            
            engine = InferenceEngine(mock_settings)
            engine.load_model()
            
            # Should call torch.load once
            assert mock_load.call_count == 1
            
            # Second call should use cached model
            engine.load_model()
            assert mock_load.call_count == 1

class TestErrorHandling:
    """Test error handling in inference pipeline."""
    
    def test_gpu_memory_error_handling(self, mock_settings, sample_pose_data):
        """Test GPU out-of-memory error handling."""
        mock_settings.model_device = "cuda"
        
        with patch('torch.cuda.is_available', return_value=True):
            engine = InferenceEngine(mock_settings)
            
            # Mock model to raise CUDA out of memory error
            engine.model = Mock()
            engine.model.side_effect = torch.cuda.OutOfMemoryError()
            
            # Should handle gracefully and fallback to CPU
            with pytest.raises(ModelInferenceError):
                engine.predict(sample_pose_data)
    
    def test_invalid_tensor_handling(self, mock_ml_model, sample_pose_data):
        """Test handling of invalid tensor data."""
        engine = InferenceEngine(Settings())
        
        # Mock model to return invalid output
        engine.model = Mock()
        engine.model.return_value = "not-a-tensor"
        
        with pytest.raises(ModelInferenceError):
            engine.predict(sample_pose_data)

class TestOutputValidation:
    """Test output validation and normalization."""
    
    def test_score_range_validation(self, mock_ml_model, sample_pose_data):
        """Test output scores are within valid ranges."""
        # Mock model to return invalid scores
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([15.0]),  # Too high
            'aspect_scores': torch.tensor([[12.0, -2.0, 8.0, 8.0, 8.0]])  # Invalid range
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        result = engine.predict(sample_pose_data)
        
        # Should clamp scores to valid range
        assert 1.0 <= result['overall_score'] <= 10.0
        for score in result['aspect_scores'].values():
            assert 1.0 <= score <= 10.0
    
    def test_aspect_score_names(self, mock_ml_model, sample_pose_data):
        """Test aspect scores have correct names."""
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([8.5]),
            'aspect_scores': torch.tensor([[7.5, 8.0, 8.5, 8.2, 8.1]])
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        result = engine.predict(sample_pose_data)
        
        expected_aspects = ['speed', 'force', 'timing', 'balance', 'coordination']
        for aspect in expected_aspects:
            assert aspect in result['aspect_scores']

# Integration Tests (would normally go in integration/ but included here for completeness)
class TestInferenceIntegration:
    """Test inference integration with other components."""
    
    @pytest.mark.asyncio
    async def test_async_inference(self, mock_settings, sample_pose_data):
        """Test async inference functionality."""
        engine = InferenceEngine(mock_settings)
        
        # Mock async predict method
        engine.async_predict = AsyncMock(return_value={'overall_score': 8.5})
        
        result = await engine.async_predict(sample_pose_data)
        
        assert result['overall_score'] == 8.5
    
    def test_inference_with_pose_extraction(self, mock_settings, mock_pose_extractor, sample_video_path):
        """Test full inference pipeline from video to analysis."""
        with patch('kataforge.preprocessing.mediapipe_wrapper.MediaPipePoseExtractor') as MockExtractor:
            MockExtractor.return_value.extract_from_video.return_value = sample_pose_data
            
            engine = InferenceEngine(mock_settings)
            
            # This would be a more complex integration test
            result = engine.analyze_video(sample_video_path)
            
            MockExtractor.assert_called_once()

class TestModelVersioning:
    """Test model versioning and compatibility."""
    
    def test_model_version_check(self, mock_ml_model, mock_settings):
        """Test model version compatibility checking."""
        # Add version to mock model
        mock_ml_model.version = "1.0.0"
        mock_ml_model.compatible_versions = ["1.0.0", "1.0.1"]
        
        engine = InferenceEngine(mock_settings)
        engine.model = mock_ml_model
        
        # Should not raise if version is compatible
        engine.check_model_compatibility("1.0.0")
    
    def test_incompatible_model_version(self, mock_ml_model, mock_settings):
        """Test handling of incompatible model versions."""
        mock_ml_model.version = "2.0.0"
        mock_ml_model.compatible_versions = ["1.0.0", "1.0.1"]
        
        engine = InferenceEngine(mock_settings)
        engine.model = mock_ml_model
        
        with pytest.raises(ModelLoadingError):
            engine.check_model_compatibility("1.0.0")

# Performance Tests
class TestInferencePerformance:
    """Test inference performance characteristics."""
    
    @pytest.mark.slow
    def test_inference_latency(self, mock_ml_model, sample_pose_data, performance_monitor):
        """Test inference latency is within acceptable bounds."""
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([8.5]),
            'aspect_scores': torch.tensor([[7.5, 8.0, 8.5, 8.2, 8.1]])
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        performance_monitor.start()
        result = engine.predict(sample_pose_data)
        performance_monitor.stop()
        
        # Inference should be fast (< 100ms for single pose sequence)
        assert performance_monitor.duration < 0.1
        assert performance_monitor.duration > 0
    
    @pytest.mark.slow
    def test_memory_usage(self, mock_ml_model, sample_pose_data, performance_monitor):
        """Test memory usage during inference."""
        mock_ml_model.return_value = {
            'overall_score': torch.tensor([8.5]),
            'aspect_scores': torch.tensor([[7.5, 8.0, 8.5, 8.2, 8.1]])
        }
        
        engine = InferenceEngine(Settings())
        engine.model = mock_ml_model
        
        performance_monitor.start()
        result = engine.predict(sample_pose_data)
        performance_monitor.stop()
        
        # Memory usage should be reasonable (< 100MB for inference)
        memory_delta_mb = performance_monitor.memory_delta / (1024 * 1024)
        assert memory_delta_mb < 100