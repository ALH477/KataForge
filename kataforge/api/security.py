"""
Security Middleware for FastAPI Application

Implements:
- Security headers (CSP, HSTS, etc.)
- Rate limiting
- Additional security measures for production
"""

from fastapi import Request, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response, JSONResponse
import time
import re
import logging

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Custom handler for security exceptions."""
    logger.warning(f"Security exception: {exc}")
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.default_headers = {
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "same-origin",
            "Permissions-Policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-site"
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Don't override existing headers
        headers_to_add = {
            k: v for k, v in self.default_headers.items()
            if k not in response.headers
        }
        
        response.headers.update(headers_to_add)
        response.headers["X-Frame-Options"] = "DENY"  # More secure default
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for basic rate limiting using token bucket algorithm."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_second = requests_per_minute / 60
        self.tokens = {}
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Use IP as a simple identifier (consider using API keys in production)
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Initialize token bucket for new clients
        if client_ip not in self.tokens:
            self.tokens[client_ip] = {
                "tokens": self.requests_per_second,
                "last_update": now
            }
        
        bucket = self.tokens[client_ip]
        
        # Calculate tokens based on elapsed time
        delta = now - bucket["last_update"]
        new_tokens = delta * self.requests_per_second
        bucket["tokens"] = min(self.requests_per_second, bucket["tokens"] + new_tokens)
        bucket["last_update"] = now
        
        # Check if request is allowed
        if bucket["tokens"] < 1:
            remaining = 1 / self.requests_per_second
            logger.warning(f"Rate limit exceeded for {client_ip}")
            response = JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
            response.headers["Retry-After"] = str(remaining)
            return response
        
        # Consume token
        bucket["tokens"] -= 1
        
        return await call_next(request)

def configure_security(app, settings):
    """Configure security middleware stack."""
    app.add_middleware(
        SecurityHeadersMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=settings.cors_allow_credentials
    )
    
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_requests
        )
    
    if settings.is_production:
        # Enforce HTTPS in production
        app.add_middleware(HTTPSRedirectMiddleware)
        
        # Remove server headers
        @app.middleware("http")
        async def remove_server_header(request: Request, call_next):
            response = await call_next(request)
            if "Server" in response.headers:
                del response.headers["Server"]
            return response
    
    return app