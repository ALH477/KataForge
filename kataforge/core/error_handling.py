"""
Centralized Error Handling System for KataForge
Provides unified error handling, recovery, and reporting across all subsystems
"""

from __future__ import annotations

import sys
import traceback
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# Error Severity and Categories
# ============================================================================

class ErrorSeverity(str, Enum):
    """Error severity levels for classification and handling"""
    CRITICAL = "critical"      # System-breaking, requires immediate attention
    HIGH = "high"              # Major functionality impaired
    MEDIUM = "medium"          # Feature degraded but system functional
    LOW = "low"                # Minor issue, no impact on core functionality
    INFO = "info"              # Informational, not an error


class ErrorCategory(str, Enum):
    """Categories for error classification"""
    # Data-related
    DATA_VALIDATION = "data_validation"
    DATA_CORRUPTION = "data_corruption"
    DATA_NOT_FOUND = "data_not_found"
    
    # Processing
    PROCESSING = "processing"
    COMPUTATION = "computation"
    TIMEOUT = "timeout"
    
    # System resources
    MEMORY = "memory"
    DISK_SPACE = "disk_space"
    GPU = "gpu"
    
    # External dependencies
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    DATABASE = "database"
    
    # ML-specific
    MODEL_LOADING = "model_loading"
    MODEL_INFERENCE = "model_inference"
    TRAINING = "training"
    
    # API/Server
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    
    # Configuration
    CONFIGURATION = "configuration"
    
    # Unknown
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Strategies for error recovery"""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"              # Use alternative method
    SKIP = "skip"                      # Skip and continue
    FAIL_FAST = "fail_fast"            # Stop immediately
    DEGRADE = "degrade"                # Continue with reduced functionality
    QUEUE = "queue"                    # Queue for later processing
    MANUAL = "manual"                  # Requires manual intervention


# ============================================================================
# Error Context and Metadata
# ============================================================================

@dataclass
class ErrorContext:
    """Rich context information about where and when an error occurred"""
    
    # Operation context
    operation: Optional[str] = None
    
    # User context
    user_id: Optional[str] = None
    
    # Data context
    input_data: Optional[Dict[str, Any]] = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metadata (generic key-value store)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Location (filled in by error handler)
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    
    # Request context (for API errors)
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    operation_id: Optional[str] = None
    
    # System context
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # Additional context
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "operation": self.operation,
            "user_id": self.user_id,
            "input_data": self._sanitize_data(self.input_data),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "file_path": self.file_path,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "operation_id": self.operation_id,
            "system_info": self.system_info,
            "extra": self.extra,
        }
    
    @staticmethod
    def _sanitize_data(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Remove sensitive data from context"""
        if not data:
            return None
        
        sensitive_keys = {'password', 'token', 'secret', 'api_key', 'authorization'}
        sanitized = {}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        
        return sanitized


# ============================================================================
# Base Exception Class
# ============================================================================

class DojoManagerError(Exception):
    """
    Base exception class for all KataForge errors.
    
    All custom exceptions should inherit from this class to ensure
    consistent error handling across the system.
    
    Attributes:
        message: Error message
        error_code: Unique error code for identification
        severity: Error severity level
        category: Error category
        recovery_strategy: Suggested recovery strategy
        context: Additional context information
        retryable: Whether the operation can be retried
        user_message: User-friendly error message
        status_code: HTTP status code (for API errors)
        details: Additional error details
    """
    
    # Default error code - subclasses should override
    _error_code: str = "KATAFORGE_ERROR"
    _default_severity: ErrorSeverity = ErrorSeverity.MEDIUM
    _default_category: ErrorCategory = ErrorCategory.UNKNOWN
    _default_recovery: RecoveryStrategy = RecoveryStrategy.FAIL_FAST
    _default_retryable: bool = False
    _default_user_message: str = (
        "Something unexpected happened. "
        "Please try again, or contact support if the problem continues."
    )
    _default_status_code: int = 500
    
    def __init__(
        self,
        message: str,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        recovery_strategy: Optional[RecoveryStrategy] = None,
        context: Optional[Union[Dict[str, Any], ErrorContext]] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
        user_message: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.severity = severity or self._default_severity
        self.category = category or self._default_category
        self.recovery_strategy = recovery_strategy or self._default_recovery
        self.cause = cause
        self.retryable = retryable if retryable is not None else self._default_retryable
        self.user_message = user_message or self._default_user_message
        self.status_code = status_code or self._default_status_code
        self.details = details or {}
        self.timestamp = datetime.now()
        
        # Handle context - can be dict or ErrorContext
        if isinstance(context, ErrorContext):
            self.context = context
        elif isinstance(context, dict):
            self.context = ErrorContext(**context) if context else self._create_default_context()
        else:
            self.context = self._create_default_context()
        
        super().__init__(self.message)
        
        # Register error in central registry
        ErrorRegistry.register(self, self.context)
    
    def _create_default_context(self) -> ErrorContext:
        """Create default context from current call stack"""
        try:
            frame = sys._getframe(3)  # Go back frames to get caller
            return ErrorContext(
                module=frame.f_globals.get('__name__', 'unknown'),
                function=frame.f_code.co_name,
                line_number=frame.f_lineno,
                file_path=frame.f_code.co_filename,
            )
        except (ValueError, AttributeError):
            return ErrorContext()
    
    @property
    def error_code(self) -> str:
        """Return the error code for this exception type."""
        return self._error_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'category': self.category.value,
            'recovery_strategy': self.recovery_strategy.value,
            'retryable': self.retryable,
            'user_message': self.user_message,
            'status_code': self.status_code,
            'details': self.details,
            'context': self.context.to_dict() if self.context else None,
            'cause': str(self.cause) if self.cause else None,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self._get_formatted_traceback(),
        }
    
    def _get_formatted_traceback(self) -> str:
        """Get formatted traceback string"""
        try:
            return ''.join(traceback.format_exception(
                type(self.cause) if self.cause else type(self),
                self.cause if self.cause else self,
                self.cause.__traceback__ if self.cause else self.__traceback__
            ))
        except Exception:
            return ""
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def should_retry(self) -> bool:
        """Determine if operation should be retried"""
        return self.retryable and self.recovery_strategy == RecoveryStrategy.RETRY
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"severity={self.severity.value}, "
            f"category={self.category.value})"
        )


# ============================================================================
# Video Processing Errors
# ============================================================================

class VideoProcessingError(DojoManagerError):
    """Errors during video preprocessing"""
    _error_code = "VIDEO_PROCESSING_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.PROCESSING
    _default_recovery = RecoveryStrategy.SKIP
    _default_user_message = (
        "We had trouble with that video. "
        "Try using MP4 format (H.264 codec), or check if the file might be damaged."
    )


class PoseExtractionError(DojoManagerError):
    """Errors during pose extraction from video"""
    _error_code = "POSE_EXTRACTION_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.PROCESSING
    _default_recovery = RecoveryStrategy.SKIP
    _default_user_message = (
        "We could not track your movements in this video. "
        "Make sure your full body is visible and the lighting is good."
    )


# ============================================================================
# Biomechanics Errors
# ============================================================================

class BiomechanicsError(DojoManagerError):
    """Errors during biomechanics calculations"""
    _error_code = "BIOMECHANICS_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_category = ErrorCategory.COMPUTATION
    _default_recovery = RecoveryStrategy.SKIP
    _default_user_message = (
        "Could not calculate the movement analysis for this video. "
        "This sometimes happens with very short clips. Try a longer recording."
    )


class ComputationError(DojoManagerError):
    """Error during computation"""
    _error_code = "COMPUTATION_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.COMPUTATION
    _default_recovery = RecoveryStrategy.FALLBACK
    _default_user_message = (
        "Something went wrong with the calculations. "
        "We are using a backup method to complete your analysis."
    )


# ============================================================================
# ML/Training Errors
# ============================================================================

class ModelTrainingError(DojoManagerError):
    """Errors during model training"""
    _error_code = "MODEL_TRAINING_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.TRAINING
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_user_message = (
        "Training ran into an issue. We will try again automatically. "
        "If this keeps happening, check that your training data is complete."
    )


class ModelInferenceError(DojoManagerError):
    """Errors during model inference"""
    _error_code = "MODEL_INFERENCE_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.MODEL_INFERENCE
    _default_recovery = RecoveryStrategy.FALLBACK
    _default_retryable = True
    _default_user_message = (
        "The AI model could not analyze this video right now. "
        "We are using a simpler analysis method as a backup."
    )


class ModelLoadingError(DojoManagerError):
    """Errors when loading trained models"""
    _error_code = "MODEL_LOADING_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_category = ErrorCategory.MODEL_LOADING
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_user_message = (
        "Could not load the AI model. "
        "Try running: kataforge model download"
    )


# ============================================================================
# Data Errors
# ============================================================================

class DataValidationError(DojoManagerError):
    """Data validation failures"""
    _error_code = "DATA_VALIDATION_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_category = ErrorCategory.DATA_VALIDATION
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_status_code = 400
    _default_user_message = (
        "Something does not look right with the input. "
        "Please check your data and try again."
    )
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if field:
            self.details['field'] = field


class DataCorruptionError(DojoManagerError):
    """Corrupted or invalid data"""
    _error_code = "DATA_CORRUPTION_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.DATA_CORRUPTION
    _default_recovery = RecoveryStrategy.SKIP
    _default_user_message = (
        "This file appears to be damaged or incomplete. "
        "Try re-exporting it from the original source."
    )


class DataNotFoundError(DojoManagerError):
    """Required data not found"""
    _error_code = "DATA_NOT_FOUND_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_category = ErrorCategory.DATA_NOT_FOUND
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_status_code = 404
    _default_user_message = (
        "We could not find what you were looking for. "
        "It may have been moved or deleted."
    )
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 resource_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if resource_type:
            self.details['resource_type'] = resource_type
        if resource_id:
            self.details['resource_id'] = resource_id


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(DojoManagerError):
    """Configuration errors"""
    _error_code = "CONFIGURATION_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_category = ErrorCategory.CONFIGURATION
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_user_message = (
        "There is a problem with the system settings. "
        "Try running: kataforge init"
    )


# ============================================================================
# System Resource Errors
# ============================================================================

class ResourceExhaustedError(DojoManagerError):
    """System resources (memory, GPU, disk) exhausted"""
    _error_code = "RESOURCE_EXHAUSTED_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_recovery = RecoveryStrategy.DEGRADE
    _default_status_code = 503
    _default_user_message = (
        "The system is currently very busy. "
        "Your request has been saved and will be processed soon."
    )
    
    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        # Map resource type to category
        category_map = {
            'memory': ErrorCategory.MEMORY,
            'gpu': ErrorCategory.GPU,
            'disk': ErrorCategory.DISK_SPACE,
        }
        kwargs.setdefault('category', category_map.get(resource_type.lower(), ErrorCategory.UNKNOWN))
        super().__init__(message, **kwargs)
        self.details['resource_type'] = resource_type


class GPUError(DojoManagerError):
    """GPU-related errors"""
    _error_code = "GPU_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_category = ErrorCategory.GPU
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_user_message = (
        "The graphics card ran into a problem. "
        "We will use the CPU instead, which may be slower."
    )


class OutOfMemoryError(DojoManagerError):
    """Out of memory error"""
    _error_code = "OUT_OF_MEMORY_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_category = ErrorCategory.MEMORY
    _default_recovery = RecoveryStrategy.DEGRADE
    _default_status_code = 503
    _default_user_message = (
        "The system needs more memory to process this. "
        "Try closing other programs or using a shorter video."
    )


class DiskSpaceError(DojoManagerError):
    """Out of disk space"""
    _error_code = "DISK_SPACE_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_category = ErrorCategory.DISK_SPACE
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_user_message = (
        "There is not enough storage space. "
        "Free up some disk space and try again."
    )


# ============================================================================
# Network Errors
# ============================================================================

class NetworkError(DojoManagerError):
    """Network connectivity errors"""
    _error_code = "NETWORK_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_category = ErrorCategory.NETWORK
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_user_message = (
        "Could not connect to the network. "
        "Check your internet connection and try again."
    )


class DatabaseError(DojoManagerError):
    """Database operation errors"""
    _error_code = "DATABASE_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.DATABASE
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_user_message = (
        "Could not save or load your data right now. "
        "Please try again in a moment."
    )


class ExternalServiceError(DojoManagerError):
    """External service error"""
    _error_code = "EXTERNAL_SERVICE_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.EXTERNAL_SERVICE
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_status_code = 503
    _default_user_message = (
        "A service we depend on is not responding. "
        "This usually resolves itself. Please try again in a few minutes."
    )
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if service_name:
            self.details['service_name'] = service_name


# ============================================================================
# API Errors
# ============================================================================

class AuthenticationError(DojoManagerError):
    """Authentication failures"""
    _error_code = "AUTHENTICATION_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.AUTHENTICATION
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_status_code = 401
    _default_user_message = (
        "We could not verify your identity. "
        "Please check your login details and try again."
    )


class AuthorizationError(DojoManagerError):
    """Authorization/permission errors"""
    _error_code = "AUTHORIZATION_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_category = ErrorCategory.AUTHORIZATION
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_status_code = 403
    _default_user_message = (
        "You do not have access to this feature. "
        "Contact your administrator if you think this is a mistake."
    )


class RateLimitError(DojoManagerError):
    """Rate limit exceeded"""
    _error_code = "RATE_LIMIT_ERROR"
    _default_severity = ErrorSeverity.LOW
    _default_category = ErrorCategory.RATE_LIMIT
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_status_code = 429
    _default_user_message = (
        "You are sending requests too quickly. "
        "Take a short break and try again."
    )
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        if retry_after:
            self.details['retry_after'] = retry_after


# ============================================================================
# Timeout Errors
# ============================================================================

class DojoTimeoutError(DojoManagerError):
    """Operation timeout"""
    _error_code = "TIMEOUT_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_category = ErrorCategory.TIMEOUT
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_status_code = 504
    _default_user_message = (
        "This is taking longer than expected. "
        "Please try again, or try with a shorter video."
    )
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        if timeout_seconds:
            self.details['timeout_seconds'] = timeout_seconds


# Backwards compatibility alias - avoid using in new code
TimeoutError = DojoTimeoutError


# ============================================================================
# Dependency Errors
# ============================================================================

class DependencyError(DojoManagerError):
    """Missing or incompatible dependencies"""
    _error_code = "DEPENDENCY_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_category = ErrorCategory.EXTERNAL_SERVICE
    _default_recovery = RecoveryStrategy.FAIL_FAST
    _default_user_message = (
        "A required component is missing. "
        "Try running: pip install kataforge[full]"
    )
    
    def __init__(self, message: str, dependency: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.details['dependency'] = dependency


# ============================================================================
# Processing Errors
# ============================================================================

class ProcessingError(DojoManagerError):
    """Generic processing error"""
    _error_code = "PROCESSING_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_category = ErrorCategory.PROCESSING
    _default_recovery = RecoveryStrategy.RETRY
    _default_retryable = True
    _default_user_message = (
        "Something went wrong while processing your request. "
        "Please try again."
    )


# ============================================================================
# Validation Error (for API compatibility)
# ============================================================================

class ValidationError(DataValidationError):
    """Alias for DataValidationError for API compatibility"""
    pass


# ============================================================================
# Error Registry for Tracking and Analytics
# ============================================================================

class ErrorRegistry:
    """
    Central registry for all errors (Singleton pattern).
    
    Tracks errors for analytics, monitoring, and debugging.
    Thread-safe implementation.
    """
    
    _instance: Optional['ErrorRegistry'] = None
    _lock = threading.Lock()
    _errors: List[Dict[str, Any]] = []
    _max_errors: int = 1000  # Keep last N errors in memory
    
    def __new__(cls) -> 'ErrorRegistry':
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    @classmethod
    def register(cls, error: Exception, context: Optional[ErrorContext] = None) -> None:
        """
        Register an error in the central registry.
        
        Args:
            error: The exception to register
            context: Optional error context
        """
        error_record = {
            'error_type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context.to_dict() if context else None,
        }
        
        # Add DojoManagerError specific fields if applicable
        if isinstance(error, DojoManagerError):
            error_record['error_code'] = error.error_code
            error_record['severity'] = error.severity.value
            error_record['category'] = error.category.value
            error_record['retryable'] = error.retryable
            error_record['status_code'] = error.status_code
        
        with cls._lock:
            cls._errors.append(error_record)
            
            # Keep only last N errors to prevent memory issues
            if len(cls._errors) > cls._max_errors:
                cls._errors = cls._errors[-cls._max_errors:]
    
    @classmethod
    def get_errors(cls, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get registered errors, optionally filtered by type.
        
        Args:
            error_type: Optional error type name to filter by
            
        Returns:
            List of error records
        """
        with cls._lock:
            if error_type is None:
                return cls._errors.copy()
            return [e for e in cls._errors if e['error_type'] == error_type]
    
    @classmethod
    def get_recent_errors(cls, count: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent errors"""
        with cls._lock:
            return cls._errors[-count:]
    
    @classmethod
    def get_errors_by_severity(cls, severity: ErrorSeverity) -> List[Dict[str, Any]]:
        """Get errors by severity level"""
        with cls._lock:
            return [e for e in cls._errors if e.get('severity') == severity.value]
    
    @classmethod
    def get_errors_by_category(cls, category: ErrorCategory) -> List[Dict[str, Any]]:
        """Get errors by category"""
        with cls._lock:
            return [e for e in cls._errors if e.get('category') == category.value]
    
    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get error statistics"""
        with cls._lock:
            total = len(cls._errors)
            if total == 0:
                return {"total_errors": 0, "by_type": {}, "by_severity": {}, "by_category": {}}
            
            by_type: Dict[str, int] = {}
            by_severity: Dict[str, int] = {}
            by_category: Dict[str, int] = {}
            
            for error in cls._errors:
                # By type
                error_type = error.get('error_type', 'Unknown')
                by_type[error_type] = by_type.get(error_type, 0) + 1
                
                # By severity (if available)
                severity = error.get('severity')
                if severity:
                    by_severity[severity] = by_severity.get(severity, 0) + 1
                
                # By category (if available)
                category = error.get('category')
                if category:
                    by_category[category] = by_category.get(category, 0) + 1
            
            return {
                "total_errors": total,
                "by_type": by_type,
                "by_severity": by_severity,
                "by_category": by_category,
            }
    
    @classmethod
    def get_error_stats(cls) -> Dict[str, Any]:
        """Get error statistics (alias for get_statistics)"""
        stats = cls.get_statistics()
        return {
            "total": stats["total_errors"],
            "by_severity": stats["by_severity"],
            "by_category": stats["by_category"],
            "by_type": stats["by_type"],
        }
    
    @classmethod
    def clear(cls) -> None:
        """Clear error registry"""
        with cls._lock:
            cls._errors.clear()
    
    @classmethod
    def export_to_file(cls, filepath: Union[str, Path]) -> None:
        """Export errors to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with cls._lock:
            errors_copy = cls._errors.copy()
        
        with open(filepath, 'w') as f:
            json.dump({
                "exported_at": datetime.now().isoformat(),
                "total_errors": len(errors_copy),
                "errors": errors_copy,
            }, f, indent=2, default=str)


# ============================================================================
# Error Handler - Decorator and Context Manager
# ============================================================================

@dataclass
class ErrorHandlerConfig:
    """Configuration for error handler"""
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    retry_exponential_backoff: bool = True
    retry_backoff_multiplier: float = 2.0
    
    # Fallback
    fallback_func: Optional[Callable[..., Any]] = None
    fallback_return_value: Any = None
    
    # Logging
    log_errors: bool = True
    log_success: bool = False
    
    # Error transformation
    transform_exception: Optional[Callable[[Exception], DojoManagerError]] = None
    
    # Recovery
    recovery_strategy: Optional[RecoveryStrategy] = None
    
    # Context
    add_context: Optional[Dict[str, Any]] = None


class ErrorHandler:
    """
    Centralized error handler with retry, fallback, and recovery logic.
    
    Can be used as decorator or context manager.
    """
    
    def __init__(self, config: Optional[ErrorHandlerConfig] = None):
        self.config = config or ErrorHandlerConfig()
        self._logger = None
    
    def __call__(self, func: F) -> F:
        """Use as decorator"""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._execute_with_handling(func, *args, **kwargs)
        return wrapper  # type: ignore
    
    def __enter__(self) -> 'ErrorHandler':
        """Use as context manager"""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Handle exception in context manager"""
        if exc_type is None:
            return False  # No exception
        
        # Transform exception to DojoManagerError
        error = self._transform_exception(exc_val)
        
        # Handle the error
        self._handle_error(error)
        
        # Suppress exception if recovery strategy says so
        return self._should_suppress_exception(error)
    
    def _execute_with_handling(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with error handling"""
        attempt = 0
        last_error: Optional[DojoManagerError] = None
        
        while attempt <= self.config.max_retries:
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success if configured
                if self.config.log_success and self._logger:
                    self._logger.info(
                        f"Successfully executed {func.__name__}",
                        extra={"function": func.__name__, "attempt": attempt}
                    )
                
                return result
                
            except DojoManagerError as e:
                last_error = e
                
                # Log error
                if self.config.log_errors:
                    self._log_error(e)
                
                # Check if should retry
                if e.should_retry() and attempt < self.config.max_retries:
                    self._wait_before_retry(attempt)
                    attempt += 1
                    continue
                
                # Check recovery strategy
                if self._should_use_fallback(e):
                    return self._execute_fallback()
                
                # Re-raise
                raise
                
            except Exception as e:
                # Transform to DojoManagerError
                dojo_error = self._transform_exception(e)
                last_error = dojo_error
                
                # Log error
                if self.config.log_errors:
                    self._log_error(dojo_error)
                
                # Check if should retry
                if dojo_error.should_retry() and attempt < self.config.max_retries:
                    self._wait_before_retry(attempt)
                    attempt += 1
                    continue
                
                # Check recovery strategy
                if self._should_use_fallback(dojo_error):
                    return self._execute_fallback()
                
                # Re-raise
                raise dojo_error from e
        
        # Max retries exceeded
        if last_error:
            raise last_error
        
        # Should never reach here, but satisfy type checker
        raise ProcessingError("Unexpected error in error handler")
    
    def _transform_exception(self, exc: Exception) -> DojoManagerError:
        """Transform exception to DojoManagerError"""
        # Already a DojoManagerError
        if isinstance(exc, DojoManagerError):
            return exc
        
        # Use custom transformer if provided
        if self.config.transform_exception:
            return self.config.transform_exception(exc)
        
        # Default transformations
        if isinstance(exc, ValueError):
            return DataValidationError(str(exc), cause=exc)
        elif isinstance(exc, FileNotFoundError):
            return DataNotFoundError(str(exc), cause=exc)
        elif isinstance(exc, MemoryError):
            return OutOfMemoryError(str(exc), cause=exc)
        elif isinstance(exc, TimeoutError):
            return DojoTimeoutError(str(exc), cause=exc)
        else:
            # Generic wrapper
            return ProcessingError(
                f"Unexpected error: {type(exc).__name__}: {exc}",
                cause=exc
            )
    
    def _handle_error(self, error: DojoManagerError) -> None:
        """Handle error according to configuration"""
        if self.config.log_errors:
            self._log_error(error)
    
    def _log_error(self, error: DojoManagerError) -> None:
        """Log error"""
        if self._logger is None:
            # Use print as fallback
            print(f"ERROR: {error}", file=sys.stderr)
            if error._get_formatted_traceback():
                print(error._get_formatted_traceback(), file=sys.stderr)
        else:
            self._logger.error(
                error.message,
                extra=error.to_dict()
            )
    
    def _should_use_fallback(self, error: DojoManagerError) -> bool:
        """Determine if should use fallback"""
        if self.config.fallback_func or self.config.fallback_return_value is not None:
            return error.recovery_strategy == RecoveryStrategy.FALLBACK
        return False
    
    def _execute_fallback(self) -> Any:
        """Execute fallback"""
        if self.config.fallback_func:
            return self.config.fallback_func()
        return self.config.fallback_return_value
    
    def _should_suppress_exception(self, error: DojoManagerError) -> bool:
        """Determine if exception should be suppressed"""
        return error.recovery_strategy in {
            RecoveryStrategy.SKIP,
            RecoveryStrategy.DEGRADE,
        }
    
    def _wait_before_retry(self, attempt: int) -> None:
        """Wait before retry with exponential backoff"""
        import time
        
        if self.config.retry_exponential_backoff:
            delay = self.config.retry_delay * (self.config.retry_backoff_multiplier ** attempt)
        else:
            delay = self.config.retry_delay
        
        time.sleep(delay)


# ============================================================================
# Convenience Functions
# ============================================================================

def handle_errors(
    max_retries: int = 3,
    fallback: Optional[Callable[..., Any]] = None,
    recovery_strategy: Optional[RecoveryStrategy] = None,
) -> Callable[[F], F]:
    """
    Decorator for error handling with retry and fallback.
    
    Example:
        @handle_errors(max_retries=3, fallback=lambda: None)
        def risky_operation():
            # may fail
            pass
    """
    config = ErrorHandlerConfig(
        max_retries=max_retries,
        fallback_func=fallback,
        recovery_strategy=recovery_strategy,
    )
    return ErrorHandler(config)


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    max_retries: int = 0,
    **kwargs: Any
) -> Optional[T]:
    """
    Safely execute a function with error handling.
    
    Returns default value on error instead of raising.
    
    Example:
        result = safe_execute(risky_func, arg1, arg2, default=None)
    """
    config = ErrorHandlerConfig(
        max_retries=max_retries,
        fallback_return_value=default,
        recovery_strategy=RecoveryStrategy.FALLBACK,
    )
    handler = ErrorHandler(config)
    
    try:
        return handler._execute_with_handling(func, *args, **kwargs)
    except DojoManagerError:
        return default


def with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff_multiplier: float = 2.0,
    **kwargs: Any
) -> T:
    """
    Execute a function with retry and exponential backoff.
    
    Args:
        func: Function to execute
        *args: Positional arguments to pass to func
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff_multiplier: Multiplier for exponential backoff
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result of the function
        
    Raises:
        The last exception if all attempts fail
        
    Example:
        result = with_retry(risky_func, arg1, max_attempts=3, delay=0.1)
    """
    import time
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Don't sleep after the last attempt
            if attempt < max_attempts - 1:
                sleep_time = delay * (backoff_multiplier ** attempt)
                time.sleep(sleep_time)
    
    # All attempts failed, raise the last exception
    if last_exception is not None:
        raise last_exception
    
    # Should never reach here
    raise ProcessingError("All retry attempts failed with no exception captured")


def create_error_context(
    operation: str,
    **kwargs: Any
) -> ErrorContext:
    """
    Create error context for manual error creation.
    
    Example:
        context = create_error_context(
            operation="video_preprocessing",
            video_path="/path/to/video.mp4"
        )
        raise ProcessingError("Failed to process video", context=context)
    """
    frame = sys._getframe(1)
    return ErrorContext(
        module=frame.f_globals.get('__name__', 'unknown'),
        function=frame.f_code.co_name,
        line_number=frame.f_lineno,
        file_path=frame.f_code.co_filename,
        operation=operation,
        extra=kwargs,
    )


# ============================================================================
# Error Reporting
# ============================================================================

class ErrorReporter:
    """Generate error reports and summaries"""
    
    @staticmethod
    def generate_summary() -> str:
        """Generate summary of recent errors"""
        stats = ErrorRegistry.get_error_stats()
        recent = ErrorRegistry.get_recent_errors(10)
        
        lines = [
            "=" * 60,
            "ERROR SUMMARY",
            "=" * 60,
            f"Total Errors: {stats.get('total', 0)}",
            "",
            "By Severity:",
        ]
        
        for severity, count in stats.get('by_severity', {}).items():
            lines.append(f"  {severity}: {count}")
        
        lines.extend([
            "",
            "By Category:",
        ])
        
        for category, count in stats.get('by_category', {}).items():
            lines.append(f"  {category}: {count}")
        
        lines.extend([
            "",
            "Recent Errors (last 10):",
            "",
        ])
        
        for error in recent:
            severity = error.get('severity', 'unknown')
            category = error.get('category', 'unknown')
            message = error.get('message', 'No message')
            lines.append(f"  [{severity}] {category}: {message}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    @staticmethod
    def export_report(filepath: Union[str, Path]) -> None:
        """Export detailed error report"""
        ErrorRegistry.export_to_file(filepath)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    # Context
    "ErrorContext",
    # Base Exception
    "DojoManagerError",
    # Video Processing
    "VideoProcessingError",
    "PoseExtractionError",
    # Biomechanics
    "BiomechanicsError",
    "ComputationError",
    # ML/Training
    "ModelTrainingError",
    "ModelInferenceError",
    "ModelLoadingError",
    # Data
    "DataValidationError",
    "DataCorruptionError",
    "DataNotFoundError",
    "ValidationError",
    # Configuration
    "ConfigurationError",
    # Resources
    "ResourceExhaustedError",
    "GPUError",
    "OutOfMemoryError",
    "DiskSpaceError",
    # Network
    "NetworkError",
    "DatabaseError",
    "ExternalServiceError",
    # API
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    # Timeout
    "DojoTimeoutError",
    "TimeoutError",
    # Dependencies
    "DependencyError",
    # Processing
    "ProcessingError",
    # Registry
    "ErrorRegistry",
    "ErrorReporter",
    # Handler
    "ErrorHandler",
    "ErrorHandlerConfig",
    # Functions
    "handle_errors",
    "safe_execute",
    "with_retry",
    "create_error_context",
]
