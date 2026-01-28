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
API Security Module

Provides:
- API key authentication
- JWT token authentication
- Rate limiting
- Security headers
- Request size limiting
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from fastapi import Depends, HTTPException, Request, Response, status
    from fastapi.security import OAuth2PasswordBearer
    from starlette.middleware.sessions import SessionMiddleware
    from starlette.datastructures import Secret
    from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    Depends = None
    HTTPException = None
    Request = None
    Response = None
    status = None
    APIKeyHeader = None
    HTTPBearer = None
    HTTPAuthorizationCredentials = None
    BaseHTTPMiddleware = object

try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    JWTError = Exception
    jwt = None

from .settings import get_settings
from .logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# API Key Authentication
# =============================================================================

class APIKeyAuth:
    """
    API Key authentication handler.
    
    Supports API keys via:
    - X-API-Key header
    - api_key query parameter
    - Authorization: ApiKey <key> header
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._valid_keys: Optional[set] = None
        self._key_hashes: Optional[set] = None
    
    @property
    def valid_keys(self) -> set:
        """Get set of valid API keys (cached)."""
        if self._valid_keys is None:
            self._valid_keys = set(self.settings.api_keys_list)
            # Also compute hashes for secure comparison
            self._key_hashes = {self._hash_key(k) for k in self._valid_keys}
        return self._valid_keys
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage/comparison."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key using constant-time comparison.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.settings.auth_enabled:
            return True
        
        if not api_key:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        key_hash = self._hash_key(api_key)
        return any(
            hmac.compare_digest(key_hash, valid_hash)
            for valid_hash in (self._key_hashes or set())
        )
    
    def extract_key_from_request(self, request: Request) -> Optional[str]:
        """
        Extract API key from request.
        
        Checks in order:
        1. X-API-Key header
        2. api_key query parameter
        3. Authorization: ApiKey <key> header
        """
        # Check header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("apikey "):
            return auth_header[7:].strip()
        
        return None
    
    async def __call__(self, request: Request) -> Optional[str]:
        """FastAPI dependency for API key authentication."""
        if not self.settings.auth_enabled:
            return None
        
        api_key = self.extract_key_from_request(request)
        
        if not api_key:
            logger.warning("auth_missing_key", path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "MISSING_API_KEY",
                    "message": "API key is required. Provide via X-API-Key header, api_key query param, or Authorization: ApiKey <key>",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        if not self.validate_key(api_key):
            logger.warning("auth_invalid_key", path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "INVALID_API_KEY",
                    "message": "Invalid API key",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        return api_key


# Create singleton instance
api_key_auth = APIKeyAuth()


def require_api_key(request: Request) -> Optional[str]:
    """Dependency that requires a valid API key."""
    return api_key_auth(request)


# =============================================================================
# JWT Authentication
# =============================================================================

class JWTAuth:
    """
    JWT token authentication handler.
    
    Supports tokens via:
    - Authorization: Bearer <token> header
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.bearer_scheme = HTTPBearer(auto_error=False) if HTTPBearer else None
    
    def create_token(
        self,
        subject: str,
        expires_delta: Optional[timedelta] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new JWT token.
        
        Args:
            subject: The subject (usually user ID or API key ID)
            expires_delta: Token expiry time (defaults to settings)
            additional_claims: Extra claims to include in token
            
        Returns:
            Encoded JWT token
        """
        if not JOSE_AVAILABLE:
            raise RuntimeError("python-jose is required for JWT authentication")
        
        if expires_delta is None:
            expires_delta = timedelta(hours=self.settings.jwt_expiry_hours)
        
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
        
        if additional_claims:
            to_encode.update(additional_claims)
        
        return jwt.encode(
            to_encode,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: The JWT token to decode
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        if not JOSE_AVAILABLE:
            raise RuntimeError("python-jose is required for JWT authentication")
        
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm],
            )
            return payload
        except JWTError as e:
            logger.warning("jwt_decode_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "INVALID_TOKEN",
                    "message": "Invalid or expired token",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)) if FASTAPI_AVAILABLE else None,
    ) -> Optional[Dict[str, Any]]:
        """FastAPI dependency for JWT authentication."""
        if not self.settings.auth_enabled:
            return None
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "MISSING_TOKEN",
                    "message": "Bearer token is required",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return self.decode_token(credentials.credentials)


# Create singleton instance
jwt_auth = JWTAuth()


# =============================================================================
# Rate Limiting
# =============================================================================

class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    
    For production with multiple workers, use Redis-based rate limiting.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._requests: Dict[str, List[float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup old entries every 60 seconds
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier from request."""
        # Use API key if available, otherwise use IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Get client IP (handle proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    def _cleanup_old_entries(self):
        """Remove expired entries to prevent memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        window = self.settings.rate_limit_window
        cutoff = now - window
        
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > cutoff
            ]
            if not self._requests[client_id]:
                del self._requests[client_id]
        
        self._last_cleanup = now
    
    def is_allowed(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            request: The incoming request
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not self.settings.rate_limit_enabled:
            return True, {}
        
        self._cleanup_old_entries()
        
        client_id = self._get_client_id(request)
        now = time.time()
        window = self.settings.rate_limit_window
        limit = self.settings.rate_limit_requests
        
        # Get requests in current window
        if client_id not in self._requests:
            self._requests[client_id] = []
        
        cutoff = now - window
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]
        
        current_count = len(self._requests[client_id])
        
        # Calculate remaining and reset time
        remaining = max(0, limit - current_count)
        reset_time = int(cutoff + window)
        
        info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "window": window,
        }
        
        if current_count >= limit:
            return False, info
        
        # Record this request
        self._requests[client_id].append(now)
        info["remaining"] = remaining - 1
        
        return True, info


# Create singleton instance
rate_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces rate limiting."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiter."""
        settings = get_settings()
        
        # Skip rate limiting for health endpoints
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        allowed, info = rate_limiter.is_allowed(request)
        
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                client=rate_limiter._get_client_id(request),
                path=request.url.path,
            )
            
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Try again in {info.get('reset', 0) - int(time.time())} seconds",
                    "retry_after": info.get("reset", 0),
                },
                headers={
                    "X-RateLimit-Limit": str(info.get("limit", 0)),
                    "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(info.get("reset", 0)),
                    "Retry-After": str(max(1, info.get("reset", 0) - int(time.time()))),
                },
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        if info:
            response.headers["X-RateLimit-Limit"] = str(info.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))
        
        return response


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to all responses."""
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        for header, value in self.SECURITY_HEADERS.items():
            if header not in response.headers:
                response.headers[header] = value
        
        # Add Content-Security-Policy for API
        if "Content-Security-Policy" not in response.headers:
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        
        return response


# =============================================================================
# Request Size Limiting Middleware
# =============================================================================

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that limits request body size."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check request size before processing."""
        settings = get_settings()
        
        # Get content length
        content_length = request.headers.get("Content-Length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > settings.max_request_size:
                    logger.warning(
                        "request_too_large",
                        size=size,
                        limit=settings.max_request_size,
                        path=request.url.path,
                    )
                    
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error_code": "REQUEST_TOO_LARGE",
                            "message": f"Request body too large. Maximum size is {settings.max_request_size} bytes",
                            "max_size": settings.max_request_size,
                            "received_size": size,
                        },
                    )
            except ValueError:
                pass
        
        return await call_next(request)


# =============================================================================
# API Key Generation Utilities
# =============================================================================

def generate_api_key(prefix: str = "dojo") -> str:
    """
    Generate a secure API key.
    
    Args:
        prefix: Prefix for the key (helps identify key type)
        
    Returns:
        A secure random API key in format: prefix_xxxxxxxxxxxxxxxx
        
    Example:
        >>> key = generate_api_key()
        >>> print(key)
        'dojo_a1b2c3d4e5f6g7h8'
    """
    random_part = secrets.token_hex(16)
    return f"{prefix}_{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.
    
    Args:
        api_key: The API key to hash
        
    Returns:
        SHA-256 hash of the key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, key_hash: str) -> bool:
    """
    Verify an API key against its hash using constant-time comparison.
    
    Args:
        api_key: The API key to verify
        key_hash: The stored hash to compare against
        
    Returns:
        True if the key matches the hash
    """
    computed_hash = hash_api_key(api_key)
    return hmac.compare_digest(computed_hash, key_hash)


# =============================================================================
# Authentication Dependency Factory
# =============================================================================

def get_auth_dependency(
    require_auth: bool = True,
    auth_type: str = "api_key",  # "api_key", "jwt", or "any"
) -> Callable:
    """
    Factory function to create authentication dependencies.
    
    Args:
        require_auth: Whether authentication is required
        auth_type: Type of authentication to use
        
    Returns:
        FastAPI dependency function
        
    Example:
        @app.get("/protected", dependencies=[Depends(get_auth_dependency())])
        async def protected_endpoint():
            pass
    """
    async def auth_dependency(request: Request):
        settings = get_settings()
        
        if not settings.auth_enabled or not require_auth:
            return None
        
        if auth_type == "api_key":
            return await api_key_auth(request)
        elif auth_type == "jwt":
            return await jwt_auth(request)
        elif auth_type == "any":
            # Try API key first, then JWT
            api_key = api_key_auth.extract_key_from_request(request)
            if api_key and api_key_auth.validate_key(api_key):
                return {"type": "api_key", "key": api_key}
            
            # Try JWT
            auth_header = request.headers.get("Authorization", "")
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()
                payload = jwt_auth.decode_token(token)
                return {"type": "jwt", "payload": payload}
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "AUTHENTICATION_REQUIRED",
                    "message": "Valid API key or JWT token required",
                },
            )
        else:
            raise ValueError(f"Unknown auth_type: {auth_type}")
    
    return auth_dependency
