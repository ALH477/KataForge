"""
Comprehensive Test Suite for Dojo Manager - Error Handling
Copyright (c) 2026 DeMoD LLC. All rights reserved.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kataforge.core.error_handling import (
    DojoManagerError,
    VideoProcessingError,
    PoseExtractionError,
    BiomechanicsError,
    ModelTrainingError,
    ModelInferenceError,
    DataValidationError,
    ConfigurationError,
    DatabaseError,
    NetworkError,
    AuthenticationError,
    AuthorizationError,
    ResourceExhaustedError,
    TimeoutError,
    DependencyError,
    handle_errors,
    safe_execute,
    with_retry,
    ErrorContext,
    ErrorRegistry
)


class TestExceptions:
    """Test all custom exceptions"""
    
    def test_base_exception(self):
        """Test DojoManagerError base exception"""
        error = DojoManagerError("Test error")
        assert str(error) == "Test error"
        assert error.error_code == "KATAFORGE_ERROR"
        assert isinstance(error, Exception)
    
    def test_video_processing_error(self):
        """Test VideoProcessingError"""
        error = VideoProcessingError("Video processing failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "VIDEO_PROCESSING_ERROR"
    
    def test_pose_extraction_error(self):
        """Test PoseExtractionError"""
        error = PoseExtractionError("Pose extraction failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "POSE_EXTRACTION_ERROR"
    
    def test_biomechanics_error(self):
        """Test BiomechanicsError"""
        error = BiomechanicsError("Biomechanics calculation failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "BIOMECHANICS_ERROR"
    
    def test_model_training_error(self):
        """Test ModelTrainingError"""
        error = ModelTrainingError("Training failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "MODEL_TRAINING_ERROR"
    
    def test_model_inference_error(self):
        """Test ModelInferenceError"""
        error = ModelInferenceError("Inference failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "MODEL_INFERENCE_ERROR"
    
    def test_data_validation_error(self):
        """Test DataValidationError"""
        error = DataValidationError("Validation failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "DATA_VALIDATION_ERROR"
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Config invalid")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "CONFIGURATION_ERROR"
    
    def test_database_error(self):
        """Test DatabaseError"""
        error = DatabaseError("Database connection failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "DATABASE_ERROR"
    
    def test_network_error(self):
        """Test NetworkError"""
        error = NetworkError("Network unreachable")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "NETWORK_ERROR"
    
    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError("Auth failed")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "AUTHENTICATION_ERROR"
    
    def test_authorization_error(self):
        """Test AuthorizationError"""
        error = AuthorizationError("Not authorized")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "AUTHORIZATION_ERROR"
    
    def test_resource_exhausted_error(self):
        """Test ResourceExhaustedError"""
        error = ResourceExhaustedError("Out of memory")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "RESOURCE_EXHAUSTED_ERROR"
    
    def test_timeout_error(self):
        """Test TimeoutError"""
        error = TimeoutError("Operation timed out")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "TIMEOUT_ERROR"
    
    def test_dependency_error(self):
        """Test DependencyError"""
        error = DependencyError("Missing dependency")
        assert isinstance(error, DojoManagerError)
        assert error.error_code == "DEPENDENCY_ERROR"


class TestErrorContext:
    """Test ErrorContext functionality"""
    
    def test_error_context_creation(self):
        """Test ErrorContext creation"""
        context = ErrorContext(
            operation="test_operation",
            user_id="test_user",
            input_data={"key": "value"}
        )
        
        assert context.operation == "test_operation"
        assert context.user_id == "test_user"
        assert context.input_data == {"key": "value"}
        assert context.timestamp is not None
    
    def test_error_context_to_dict(self):
        """Test ErrorContext serialization"""
        context = ErrorContext(
            operation="test",
            metadata={"test": "data"}
        )
        
        context_dict = context.to_dict()
        assert "operation" in context_dict
        assert "timestamp" in context_dict
        assert "metadata" in context_dict
        assert context_dict["metadata"]["test"] == "data"


class TestHandleErrors:
    """Test handle_errors decorator"""
    
    def test_handle_errors_success(self):
        """Test successful execution with decorator"""
        @handle_errors()
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_handle_errors_with_exception(self):
        """Test exception handling with decorator"""
        @handle_errors()
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
    
    def test_handle_errors_with_retry(self):
        """Test retry mechanism"""
        call_count = 0
        
        @handle_errors(max_retries=3)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Network error")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3
    
    def test_handle_errors_max_retries_exceeded(self):
        """Test max retries exceeded"""
        @handle_errors(max_retries=2)
        def always_fails():
            raise NetworkError("Always fails")
        
        with pytest.raises(NetworkError):
            always_fails()
    
    def test_handle_errors_with_fallback(self):
        """Test fallback execution"""
        def fallback_func():
            return "fallback"
        
        @handle_errors(fallback=fallback_func)
        def failing_func():
            raise Exception("Error")
        
        result = failing_func()
        assert result == "fallback"


class TestSafeExecute:
    """Test safe_execute function"""
    
    def test_safe_execute_success(self):
        """Test successful execution"""
        def success_func():
            return "success"
        
        result = safe_execute(success_func)
        assert result == "success"
    
    def test_safe_execute_with_exception(self):
        """Test exception handling"""
        def failing_func():
            raise ValueError("Error")
        
        result = safe_execute(failing_func, default="default")
        assert result == "default"
    
    def test_safe_execute_with_args(self):
        """Test with arguments"""
        def add(a, b):
            return a + b
        
        result = safe_execute(add, 5, 3)
        assert result == 8
    
    def test_safe_execute_with_kwargs(self):
        """Test with keyword arguments"""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = safe_execute(greet, name="World", greeting="Hi")
        assert result == "Hi, World!"


class TestWithRetry:
    """Test with_retry function"""
    
    def test_with_retry_success(self):
        """Test successful retry"""
        call_count = 0
        
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Flaky")
            return "success"
        
        result = with_retry(flaky, max_attempts=3)
        assert result == "success"
        assert call_count == 2
    
    def test_with_retry_all_attempts_failed(self):
        """Test all retry attempts failed"""
        def always_fails():
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            with_retry(always_fails, max_attempts=3)
    
    def test_with_retry_exponential_backoff(self):
        """Test exponential backoff"""
        import time
        
        call_times = []
        
        def record_time():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Not yet")
            return "success"
        
        result = with_retry(record_time, max_attempts=3, delay=0.1)
        assert result == "success"
        
        # Check that delays increase
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1


class TestErrorRegistry:
    """Test ErrorRegistry functionality"""
    
    def test_error_registry_singleton(self):
        """Test ErrorRegistry is singleton"""
        registry1 = ErrorRegistry()
        registry2 = ErrorRegistry()
        assert registry1 is registry2
    
    def test_register_error(self):
        """Test error registration"""
        registry = ErrorRegistry()
        error = ValueError("Test error")
        context = ErrorContext(operation="test")
        
        registry.register(error, context)
        
        errors = registry.get_errors()
        assert len(errors) > 0
        assert errors[-1]['error_type'] == 'ValueError'
        assert errors[-1]['message'] == 'Test error'
    
    def test_get_errors_by_type(self):
        """Test filtering errors by type"""
        registry = ErrorRegistry()
        registry.clear()  # Clear previous errors
        
        error1 = ValueError("Error 1")
        error2 = TypeError("Error 2")
        error3 = ValueError("Error 3")
        
        context = ErrorContext(operation="test")
        
        registry.register(error1, context)
        registry.register(error2, context)
        registry.register(error3, context)
        
        value_errors = registry.get_errors(error_type='ValueError')
        assert len(value_errors) >= 2
        
        type_errors = registry.get_errors(error_type='TypeError')
        assert len(type_errors) >= 1
    
    def test_get_statistics(self):
        """Test error statistics"""
        registry = ErrorRegistry()
        registry.clear()
        
        # Register some errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            context = ErrorContext(operation="test")
            registry.register(error, context)
        
        stats = registry.get_statistics()
        assert stats['total_errors'] >= 5
        assert 'ValueError' in stats['by_type']
    
    def test_clear_errors(self):
        """Test clearing error registry"""
        registry = ErrorRegistry()
        
        # Add some errors
        error = ValueError("Test")
        context = ErrorContext(operation="test")
        registry.register(error, context)
        
        # Clear
        registry.clear()
        
        # Check cleared
        errors = registry.get_errors()
        assert len(errors) == 0


class TestIntegration:
    """Integration tests for error handling"""
    
    def test_end_to_end_error_flow(self):
        """Test complete error handling flow"""
        registry = ErrorRegistry()
        initial_count = len(registry.get_errors())
        
        @handle_errors(max_retries=2)
        def process_data(data):
            if not isinstance(data, dict):
                raise DataValidationError("Data must be dict")
            return data.get('value', 0) * 2
        
        # Test success
        result = process_data({'value': 5})
        assert result == 10
        
        # Test failure
        with pytest.raises(DataValidationError):
            process_data("invalid")
        
        # Check error was registered
        errors = registry.get_errors()
        assert len(errors) > initial_count
    
    def test_nested_error_handling(self):
        """Test nested error handlers"""
        @handle_errors()
        def outer_func():
            @handle_errors()
            def inner_func():
                raise ValueError("Inner error")
            
            return inner_func()
        
        with pytest.raises(ValueError):
            outer_func()
    
    def test_error_context_propagation(self):
        """Test error context propagates through call stack"""
        @handle_errors()
        def level3():
            raise RuntimeError("Level 3 error")
        
        @handle_errors()
        def level2():
            return level3()
        
        @handle_errors()
        def level1():
            return level2()
        
        with pytest.raises(RuntimeError):
            level1()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
