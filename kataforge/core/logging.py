"""
KataForge - Adaptive Martial Arts Analysis System
Copyright © 2026 DeMoD LLC. All rights reserved.

This file is part of KataForge, released under the KataForge License
(based on Elastic License v2). See LICENSE in the project root for full terms.

SPDX-License-Identifier: Elastic-2.0

Description:
    [Brief module description – please edit]

Usage notes:
    - Private self-hosting, dojo use, and modifications are permitted.
    - Offering as a hosted/managed service to third parties is prohibited
      without explicit written permission from DeMoD LLC.
"""

"""
Structured Logging Configuration

Provides production-ready logging with:
- JSON output for production (machine-parseable)
- Colored console output for development (human-readable)
- Request ID and context binding
- Automatic exception formatting
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

try:
    import structlog
    from structlog.types import Processor
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

from .settings import get_settings

# Context variable for request-scoped data (request ID, user, etc.)
request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    ctx = request_context.get()
    return ctx.get("request_id")


def set_request_context(**kwargs) -> None:
    """Set request context variables."""
    ctx = request_context.get().copy()
    ctx.update(kwargs)
    request_context.set(ctx)


def clear_request_context() -> None:
    """Clear request context (call at end of request)."""
    request_context.set({})


def add_request_context(
    logger: logging.Logger, 
    method_name: str, 
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Structlog processor to add request context to all log entries."""
    ctx = request_context.get()
    if ctx:
        event_dict.update(ctx)
    return event_dict


def add_app_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add application-level context to log entries."""
    settings = get_settings()
    event_dict["app"] = settings.app_name
    event_dict["environment"] = settings.environment.value if hasattr(settings.environment, 'value') else settings.environment
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    Call this once at application startup before any logging occurs.
    
    Example:
        >>> from kataforge.core.logging import setup_logging, get_logger
        >>> setup_logging()
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started", version="1.0.0")
    """
    settings = get_settings()
    
    # Determine log level
    log_level = getattr(logging, settings.log_level if isinstance(settings.log_level, str) else settings.log_level.value)
    
    # Determine output format
    use_json = settings.effective_log_format == "json"
    
    if STRUCTLOG_AVAILABLE:
        _setup_structlog(log_level, use_json)
    else:
        _setup_stdlib_logging(log_level, use_json)


def _setup_structlog(log_level: int, use_json: bool) -> None:
    """Configure structlog."""
    
    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_request_context,
        add_app_context,
    ]
    
    if use_json:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Colored console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Set level for common noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(log_level)


def _setup_stdlib_logging(log_level: int, use_json: bool) -> None:
    """Fallback to stdlib logging when structlog is not available."""
    
    if use_json:
        # Simple JSON-like format
        log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    else:
        # Human-readable format
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logging.basicConfig(
        format=log_format,
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def get_logger(name: Optional[str] = None) -> Any:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__). If None, returns root logger.
        
    Returns:
        Logger instance (structlog if available, stdlib otherwise)
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", file="video.mp4", frames=100)
        >>> logger.error("Processing failed", error="file not found", exc_info=True)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


class LoggerAdapter:
    """
    Adapter that provides a consistent interface regardless of whether
    structlog is available.
    """
    
    def __init__(self, name: Optional[str] = None):
        self._logger = get_logger(name)
        self._structlog = STRUCTLOG_AVAILABLE
    
    def _log(self, level: str, msg: str, **kwargs):
        """Internal logging method."""
        if self._structlog:
            getattr(self._logger, level)(msg, **kwargs)
        else:
            # For stdlib logging, format kwargs into message
            if kwargs:
                extra_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                msg = f"{msg} | {extra_str}"
            getattr(self._logger, level)(msg)
    
    def debug(self, msg: str, **kwargs):
        self._log("debug", msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log("info", msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log("warning", msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log("error", msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log("critical", msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log an exception with traceback."""
        kwargs["exc_info"] = True
        self._log("error", msg, **kwargs)
    
    def bind(self, **kwargs) -> "LoggerAdapter":
        """Bind context to logger (structlog only, no-op for stdlib)."""
        if self._structlog:
            self._logger = self._logger.bind(**kwargs)
        return self
