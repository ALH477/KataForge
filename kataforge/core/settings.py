"""
KataForge Application Settings with Environment Variable Support

Configuration is loaded in this priority order:
1. Environment variables (highest priority)
2. .env file
3. Default values (lowest priority)

All environment variables are prefixed with DOJO_ (e.g., DOJO_API_HOST)

Quick start: Run `kataforge init` to set up your training space.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from pydantic import Field, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseSettings = object
    SettingsConfigDict = None
    Field = lambda *args, **kwargs: None
    model_validator = lambda *args, **kwargs: lambda f: f


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    JSON = "json"
    CONSOLE = "console"
    AUTO = "auto"  # JSON in production, console in development


if PYDANTIC_AVAILABLE:
    class Settings(BaseSettings):
        """
        KataForge application settings loaded from environment variables.
        
        You can override any setting with environment variables prefixed with DOJO_.
        For example: export DOJO_API_PORT=8000
        
        Most settings work well with their defaults. Only change them if you need
        specific behavior for your training environment.
        """
        
        model_config = SettingsConfigDict(
            env_prefix="DOJO_",
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
        
        # ===========================================
        # Application
        # ===========================================
        
        app_name: str = Field(
            default="kataforge",
            description="Application name for logging and identification"
        )
        
        environment: Environment = Field(
            default=Environment.DEVELOPMENT,
            description="Application environment (development, staging, production, testing)"
        )
        
        debug: bool = Field(
            default=False,
            description="Enable debug mode (more verbose logging, detailed errors)"
        )
        
        # ===========================================
        # API Server
        # ===========================================
        
        api_host: str = Field(
            default="0.0.0.0",
            description="Host to bind API server to"
        )
        
        api_port: int = Field(
            default=8000,
            ge=1,
            le=65535,
            description="Port for API server"
        )
        
        api_workers: int = Field(
            default=1,
            ge=1,
            le=32,
            description="Number of API worker processes"
        )
        
        api_reload: bool = Field(
            default=False,
            description="Enable auto-reload on code changes (development only)"
        )
        
        # ===========================================
        # Model & Inference
        # ===========================================
        
        model_path: Optional[str] = Field(
            default=None,
            description="Path to trained model file"
        )
        
        model_dir: str = Field(
            default="~/.kataforge/models",
            description="Directory for model files"
        )
        
        model_device: str = Field(
            default="auto",
            description="Device for model inference (auto, cpu, cuda, rocm)"
        )
        
        # ===========================================
        # Data & Storage
        # ===========================================
        
        data_dir: str = Field(
            default="~/.kataforge/data",
            description="Base directory for data storage (profiles, models, poses)"
        )
        
        # ===========================================
        # Logging
        # ===========================================
        
        log_level: LogLevel = Field(
            default=LogLevel.INFO,
            description="Minimum log level"
        )
        
        log_format: LogFormat = Field(
            default=LogFormat.AUTO,
            description="Log output format (json, console, auto)"
        )
        
        log_file: Optional[str] = Field(
            default=None,
            description="Path to log file (if not set, logs to stdout)"
        )
        
        # ===========================================
        # Timeouts & Limits
        # ===========================================
        
        request_timeout: int = Field(
            default=30,
            ge=1,
            le=300,
            description="Request timeout in seconds"
        )
        
        max_request_size: int = Field(
            default=100 * 1024 * 1024,  # 100MB
            description="Maximum request body size in bytes"
        )
        
        shutdown_timeout: int = Field(
            default=30,
            ge=1,
            le=300,
            description="Graceful shutdown timeout in seconds"
        )
        
        # ===========================================
        # GPU / Hardware
        # ===========================================
        
        gpu_memory_fraction: float = Field(
            default=0.9,
            ge=0.1,
            le=1.0,
            description="Fraction of GPU memory to use"
        )
        
        rocm_arch: Optional[str] = Field(
            default=None,
            description="ROCm GPU architecture (e.g., gfx1100 for RDNA3)"
        )
        
        # ===========================================
        # Feature Flags
        # ===========================================
        
        enable_metrics: bool = Field(
            default=True,
            description="Enable Prometheus metrics endpoint"
        )
        
        enable_tracing: bool = Field(
            default=False,
            description="Enable OpenTelemetry tracing"
        )
        
        # TLS/SSL Configuration
        tls_enabled: bool = Field(
            default=False,
            description="Enable TLS/SSL for the API server. Should be handled by reverse proxy in production"
        )
        
        tls_cert_file: Optional[str] = Field(
            default=None,
            description="Path to TLS certificate file"
        )
        
        tls_key_file: Optional[str] = Field(
            default=None,
            description="Path to TLS private key file"
        )
        
        auth_enabled: bool = Field(
            default=False,
            description="Enable API key authentication (disable for development)"
        )
        
        api_keys: str = Field(
            default="",
            description="Comma-separated list of valid API keys (or use DOJO_API_KEYS_FILE)"
        )
        
        api_keys_file: Optional[str] = Field(
            default=None,
            description="Path to file containing API keys (one per line)"
        )
        
        jwt_secret_key: str = Field(
            default="CHANGE-ME-IN-PRODUCTION",
            description="Secret key for JWT signing (MUST change in production)"
        )
        
        jwt_algorithm: str = Field(
            default="HS256",
            description="JWT signing algorithm"
        )
        
        jwt_expiry_hours: int = Field(
            default=24,
            ge=1,
            le=720,
            description="JWT token expiry time in hours"
        )
        
        # ===========================================
        # Rate Limiting
        # ===========================================
        
        rate_limit_enabled: bool = Field(
            default=True,
            description="Enable rate limiting"
        )
        
        rate_limit_requests: int = Field(
            default=100,
            ge=1,
            description="Maximum requests per time window"
        )
        
        rate_limit_window: int = Field(
            default=60,
            ge=1,
            description="Rate limit time window in seconds"
        )
        
        rate_limit_burst: int = Field(
            default=20,
            ge=1,
            description="Burst allowance above normal rate"
        )
        
        # ===========================================
        # CORS
        # ===========================================
        
        cors_origins: str = Field(
            default="",
            description="Comma-separated list of allowed CORS origins (empty = none, * = all)"
        )
        
        cors_allow_credentials: bool = Field(
            default=False,
            description="Allow credentials in CORS requests"
        )
        
        # ===========================================
        # Gradio UI
        # ===========================================
        
        gradio_host: str = Field(
            default="0.0.0.0",
            description="Gradio server host"
        )
        
        gradio_port: int = Field(
            default=7860,
            ge=1,
            le=65535,
            description="Gradio server port"
        )
        
        gradio_share: bool = Field(
            default=False,
            description="Create public Gradio share link"
        )
        
        # ===========================================
        # Ollama / LLM
        # ===========================================
        
        ollama_host: str = Field(
            default="http://localhost:11434",
            description="Ollama API URL"
        )
        
        ollama_vision_model: str = Field(
            default="llava:7b",
            description="Vision model for frame analysis"
        )
        
        ollama_text_model: str = Field(
            default="mistral:7b",
            description="Text model for feedback generation"
        )
        
        ollama_timeout: int = Field(
            default=120,
            ge=10,
            le=600,
            description="Ollama request timeout in seconds"
        )
        
        llm_backend: str = Field(
            default="ollama",
            description="LLM backend: ollama or llamacpp"
        )
        
        # ===========================================
        # Voice / TTS / STT
        # ===========================================
        
        tts_enabled: bool = Field(
            default=True,
            description="Enable text-to-speech"
        )
        
        tts_provider: str = Field(
            default="piper",
            description="TTS provider: piper, coqui, edge, browser"
        )
        
        tts_voice: str = Field(
            default="en_US-lessac-medium",
            description="TTS voice model"
        )
        
        tts_speed: float = Field(
            default=1.0,
            ge=0.5,
            le=2.0,
            description="Speech rate"
        )
        
        stt_enabled: bool = Field(
            default=True,
            description="Enable speech-to-text"
        )
        
        stt_provider: str = Field(
            default="whisper",
            description="STT provider: whisper, browser"
        )
        
        stt_model: str = Field(
            default="base",
            description="Whisper model size"
        )
        
        stt_language: str = Field(
            default="en",
            description="Recognition language"
        )
        
        voice_activation_phrase: str = Field(
            default="hey coach",
            description="Wake word for voice activation"
        )
        
        voice_feedback_auto_play: bool = Field(
            default=True,
            description="Auto-play TTS feedback"
        )
        
        # ===========================================
        # Validators
        # ===========================================
        
        @model_validator(mode='after')
        def validate_production_settings(self) -> 'Settings':
            """Validate security requirements for production environments.
            
            In production environments, we enforce stricter security policies:
            - Strong JWT secret
            - Authentication mandatory
            - Debug mode disabled
            - TLS required for public endpoints
            - Rate limiting enabled
            """
            if self.is_production:
                # JWT Secret Validation
                weak_secrets = ['CHANGE-ME-IN-PRODUCTION', '', 'secret', 'dev-secret']
                if self.jwt_secret_key in weak_secrets or len(self.jwt_secret_key) < 32:
                    raise ValueError(
                        "Critical: JWT_SECRET_KEY must be a strong, randomly generated string of at least 64 characters. "
                        "Do NOT use common passwords or easily guessable strings. "
                        "Run: openssl rand -hex 32 to generate a secure key."
                    )
                
                # Authentication Enforcement
                if not self.auth_enabled:
                    raise ValueError("API authentication must be enabled in production (set AUTH_ENABLED=true)")
                
                # Debug Mode Check
                if self.debug:
                    raise ValueError("Debug mode must be disabled in production (set DEBUG=false)")
                
                # Rate Limiting Enforcement
                if not self.rate_limit_enabled:
                    raise ValueError("Rate limiting must be enabled in production (set RATE_LIMIT_ENABLED=true)")
                
                # TLS Requirements for Public Endpoints
                if self.api_host not in ['127.0.0.1', 'localhost', '::1']:
                    if not hasattr(self, 'tls_enabled') or not self.tls_enabled:
                        raise ValueError(
                            "TLS/SSL must be enabled for production API endpoints. "
                            "Configure reverse proxy with Let's Encrypt or equivalent."
                        )
                
                # Check for default credentials
                if 'admin:admin' in self.api_keys_list or 'test:test' in self.api_keys_list:
                    raise ValueError("Default credentials are not allowed in production. Please change API keys.")
            
            return self
        
        # ===========================================
        # Computed Properties
        # ===========================================
        
        @property
        def is_production(self) -> bool:
            """Check if running in production."""
            return self.environment == Environment.PRODUCTION
        
        @property
        def is_development(self) -> bool:
            """Check if running in development."""
            return self.environment == Environment.DEVELOPMENT
        
        @property
        def effective_log_format(self) -> str:
            """Get the effective log format based on environment."""
            if self.log_format == LogFormat.AUTO:
                return "json" if self.is_production else "console"
            return self.log_format.value
        
        @property
        def cors_origins_list(self) -> list:
            """Parse CORS origins into a list."""
            if not self.cors_origins:
                return []
            if self.cors_origins == "*":
                return ["*"]
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        
        @property
        def api_keys_list(self) -> list:
            """Get API keys from string and/or file."""
            keys = []
            if self.api_keys:
                keys.extend([k.strip() for k in self.api_keys.split(",") if k.strip()])
            if self.api_keys_file:
                try:
                    with open(self.api_keys_file) as f:
                        keys.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])
                except Exception:
                    pass
            return keys
        
        @property
        def resolved_model_dir(self) -> Path:
            """Get resolved model directory path."""
            return Path(self.model_dir).expanduser()
        
        @property
        def resolved_data_dir(self) -> Path:
            """Get resolved data directory path."""
            return Path(self.data_dir).expanduser()

else:
    # Fallback for when pydantic-settings is not available
    class Settings:
        """Fallback settings when pydantic-settings is not available."""
        
        def __init__(self):
            # Application settings
            self.app_name = os.getenv("DOJO_APP_NAME", "kataforge")
            self.environment = os.getenv("DOJO_ENVIRONMENT", "development")
            self.debug = os.getenv("DOJO_DEBUG", "false").lower() == "true"
            
            # API settings
            self.api_host = os.getenv("DOJO_API_HOST", "0.0.0.0")
            self.api_port = int(os.getenv("DOJO_API_PORT", "8000"))
            self.api_workers = int(os.getenv("DOJO_API_WORKERS", "1"))
            self.api_reload = os.getenv("DOJO_API_RELOAD", "false").lower() == "true"
            
            # Model settings
            self.model_path = os.getenv("DOJO_MODEL_PATH")
            self.model_dir = os.getenv("DOJO_MODEL_DIR", "~/.kataforge/models")
            self.model_device = os.getenv("DOJO_MODEL_DEVICE", "auto")
            
            # Data settings
            self.data_dir = os.getenv("DOJO_DATA_DIR", "~/.kataforge/data")
            
            # Logging settings
            self.log_level = os.getenv("DOJO_LOG_LEVEL", "INFO")
            self.log_format = os.getenv("DOJO_LOG_FORMAT", "auto")
            self.log_file = os.getenv("DOJO_LOG_FILE")
            
            # Timeout settings
            self.request_timeout = int(os.getenv("DOJO_REQUEST_TIMEOUT", "30"))
            self.max_request_size = int(os.getenv("DOJO_MAX_REQUEST_SIZE", str(100 * 1024 * 1024)))
            self.shutdown_timeout = int(os.getenv("DOJO_SHUTDOWN_TIMEOUT", "30"))
            
            # GPU settings
            self.gpu_memory_fraction = float(os.getenv("DOJO_GPU_MEMORY_FRACTION", "0.9"))
            self.rocm_arch = os.getenv("DOJO_ROCM_ARCH")
            
            # Feature flags
            self.enable_metrics = os.getenv("DOJO_ENABLE_METRICS", "true").lower() == "true"
            self.enable_tracing = os.getenv("DOJO_ENABLE_TRACING", "false").lower() == "true"
            
            # Security settings
            self.auth_enabled = os.getenv("DOJO_AUTH_ENABLED", "false").lower() == "true"
            self.api_keys = os.getenv("DOJO_API_KEYS", "")
            self.api_keys_file = os.getenv("DOJO_API_KEYS_FILE")
            self.jwt_secret_key = os.getenv("DOJO_JWT_SECRET_KEY", "CHANGE-ME-IN-PRODUCTION")
            self.jwt_algorithm = os.getenv("DOJO_JWT_ALGORITHM", "HS256")
            self.jwt_expiry_hours = int(os.getenv("DOJO_JWT_EXPIRY_HOURS", "24"))
            
            # Rate limiting settings
            self.rate_limit_enabled = os.getenv("DOJO_RATE_LIMIT_ENABLED", "true").lower() == "true"
            self.rate_limit_requests = int(os.getenv("DOJO_RATE_LIMIT_REQUESTS", "100"))
            self.rate_limit_window = int(os.getenv("DOJO_RATE_LIMIT_WINDOW", "60"))
            self.rate_limit_burst = int(os.getenv("DOJO_RATE_LIMIT_BURST", "20"))
            
            # CORS settings
            self.cors_origins = os.getenv("DOJO_CORS_ORIGINS", "")
            self.cors_allow_credentials = os.getenv("DOJO_CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
            
            # Gradio UI settings
            self.gradio_host = os.getenv("DOJO_GRADIO_HOST", "0.0.0.0")
            self.gradio_port = int(os.getenv("DOJO_GRADIO_PORT", "7860"))
            self.gradio_share = os.getenv("DOJO_GRADIO_SHARE", "false").lower() == "true"
            
            # Ollama / LLM settings
            self.ollama_host = os.getenv("DOJO_OLLAMA_HOST", "http://localhost:11434")
            self.ollama_vision_model = os.getenv("DOJO_OLLAMA_VISION_MODEL", "llava:7b")
            self.ollama_text_model = os.getenv("DOJO_OLLAMA_TEXT_MODEL", "mistral:7b")
            self.ollama_timeout = int(os.getenv("DOJO_OLLAMA_TIMEOUT", "120"))
            self.llm_backend = os.getenv("DOJO_LLM_BACKEND", "ollama")
            
            # Voice settings
            self.tts_enabled = os.getenv("DOJO_TTS_ENABLED", "true").lower() == "true"
            self.tts_provider = os.getenv("DOJO_TTS_PROVIDER", "piper")
            self.tts_voice = os.getenv("DOJO_TTS_VOICE", "en_US-lessac-medium")
            self.tts_speed = float(os.getenv("DOJO_TTS_SPEED", "1.0"))
            self.stt_enabled = os.getenv("DOJO_STT_ENABLED", "true").lower() == "true"
            self.stt_provider = os.getenv("DOJO_STT_PROVIDER", "whisper")
            self.stt_model = os.getenv("DOJO_STT_MODEL", "base")
            self.stt_language = os.getenv("DOJO_STT_LANGUAGE", "en")
            self.voice_activation_phrase = os.getenv("DOJO_VOICE_ACTIVATION_PHRASE", "hey coach")
            self.voice_feedback_auto_play = os.getenv("DOJO_VOICE_FEEDBACK_AUTO_PLAY", "true").lower() == "true"
            
            # Validate production settings
            if self.environment == 'production':
                if self.jwt_secret_key == 'CHANGE-ME-IN-PRODUCTION':
                    raise ValueError(
                        "For production use, please set DOJO_JWT_SECRET_KEY to a secure random string. "
                        "Run: kataforge init to get started"
                    )
                if not self.auth_enabled:
                    raise ValueError(
                        "For production, please enable API authentication with DOJO_AUTH_ENABLED=true"
                    )
        
        @property
        def is_production(self) -> bool:
            return self.environment == "production"
        
        @property
        def is_development(self) -> bool:
            return self.environment == "development"
        
        @property
        def effective_log_format(self) -> str:
            if self.log_format == "auto":
                return "json" if self.is_production else "console"
            return self.log_format
        
        @property
        def cors_origins_list(self) -> list:
            if not self.cors_origins:
                return []
            if self.cors_origins == "*":
                return ["*"]
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        
        @property
        def api_keys_list(self) -> list:
            keys = []
            if self.api_keys:
                keys.extend([k.strip() for k in self.api_keys.split(",") if k.strip()])
            if self.api_keys_file:
                try:
                    with open(self.api_keys_file) as f:
                        keys.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])
                except Exception:
                    pass
            return keys
        
        @property
        def resolved_model_dir(self) -> Path:
            """Get resolved model directory path."""
            return Path(self.model_dir).expanduser()
        
        @property
        def resolved_data_dir(self) -> Path:
            """Get resolved data directory path."""
            return Path(self.data_dir).expanduser()


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached singleton).
    
    Returns:
        Settings instance loaded from environment
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.api_port)
        8000
    """
    return Settings()


# Convenience function for testing - allows clearing the cache
def clear_settings_cache():
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
