"""
KataForge Core Module

Provides shared utilities, configuration, logging, and error handling.
"""

from .settings import Settings, get_settings, clear_settings_cache, Environment, LogLevel, LogFormat
from .logging import setup_logging, get_logger, LoggerAdapter, set_request_context, clear_request_context
from .error_handling import (
    # Enums
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    # Context
    ErrorContext,
    # Base Exception
    DojoManagerError,
    # Video Processing
    VideoProcessingError,
    PoseExtractionError,
    # Biomechanics
    BiomechanicsError,
    ComputationError,
    # ML/Training
    ModelTrainingError,
    ModelInferenceError,
    ModelLoadingError,
    # Data
    DataValidationError,
    DataCorruptionError,
    DataNotFoundError,
    ValidationError,
    # Configuration
    ConfigurationError,
    # Resources
    ResourceExhaustedError,
    GPUError,
    OutOfMemoryError,
    DiskSpaceError,
    # Network
    NetworkError,
    DatabaseError,
    ExternalServiceError,
    # API
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    # Timeout
    DojoTimeoutError,
    # Dependencies
    DependencyError,
    # Processing
    ProcessingError,
    # Registry
    ErrorRegistry,
    ErrorReporter,
    # Handler
    ErrorHandler,
    ErrorHandlerConfig,
    # Functions
    handle_errors,
    safe_execute,
    with_retry,
    create_error_context,
)

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "clear_settings_cache",
    "Environment",
    "LogLevel",
    "LogFormat",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    "set_request_context",
    "clear_request_context",
    # Error Enums
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    # Error Context
    "ErrorContext",
    # Errors
    "DojoManagerError",
    "VideoProcessingError",
    "PoseExtractionError",
    "BiomechanicsError",
    "ComputationError",
    "ModelTrainingError",
    "ModelInferenceError",
    "ModelLoadingError",
    "DataValidationError",
    "DataCorruptionError",
    "DataNotFoundError",
    "ValidationError",
    "ConfigurationError",
    "ResourceExhaustedError",
    "GPUError",
    "OutOfMemoryError",
    "DiskSpaceError",
    "NetworkError",
    "DatabaseError",
    "ExternalServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "DojoTimeoutError",
    "DependencyError",
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
