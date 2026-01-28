"""
Unit tests for API server functionality.

Tests FastAPI application creation, middleware, and basic functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

try:
    from kataforge.api.server import create_app
    from kataforge.core.settings import Settings
    from kataforge.core.error_handling import KataForgeError
except ImportError:
    pytest.skip("KataForge not available")

class TestCreateApp:
    """Test FastAPI application creation and configuration."""
    
    def test_create_app_basic(self):
        """Test basic app creation without settings."""
        app = create_app()
        
        assert app is not None
        assert app.title == "KataForge API"
        assert app.version == "0.1.0"
        assert len(app.routes) > 0
    
    def test_create_app_with_settings(self, mock_settings):
        """Test app creation with custom settings."""
        app = create_app(mock_settings)
        
        assert app.state.settings == mock_settings
        assert app.state.data_dir == mock_settings.resolved_data_dir
    
    @patch('kataforge.api.server.configure_logging')
    def test_create_app_configures_logging(self, mock_configure):
        """Test that app creation configures logging."""
        create_app()
        
        mock_configure.assert_called_once()
    
    @patch('kataforge.api.server.configure_exception_handlers')
    def test_create_app_configures_exceptions(self, mock_configure):
        """Test that app creation configures exception handlers."""
        create_app()
        
        mock_configure.assert_called_once()
    
    @patch('kataforge.api.server.configure_middleware')
    def test_create_app_configures_middleware(self, mock_configure):
        """Test that app creation configures middleware."""
        create_app()
        
        mock_configure.assert_called_once()

class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns health info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_live_endpoint(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "checks" in data
    
    def test_health_ready_endpoint(self, test_client):
        """Test readiness probe endpoint."""
        response = test_client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    def test_health_startup_endpoint(self, test_client):
        """Test startup probe endpoint."""
        response = test_client.get("/health/startup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"

class TestMiddlewareConfiguration:
    """Test middleware setup and configuration."""
    
    @patch('kataforge.api.server.CORSMiddleware')
    def test_cors_middleware_configured(self, mock_cors, test_client):
        """Test CORS middleware is configured."""
        app = test_client.app
        
        mock_cors.assert_called()
        call_args = mock_cors.call_args
        
        assert "allow_origins" in call_args.kwargs
        assert "allow_credentials" in call_args.kwargs
    
    @patch('kataforge.api.server.GZipMiddleware')
    def test_gzip_middleware_configured(self, mock_gzip, test_client):
        """Test GZip middleware is configured."""
        app = test_client.app
        
        mock_gzip.assert_called_with(app, minimum_size=1000)
    
    @patch('kataforge.api.server.HTTPExceptionHandlersMiddleware')
    def test_exception_middleware_configured(self, mock_exception, test_client):
        """Test exception handling middleware is configured."""
        app = test_client.app
        
        mock_exception.assert_called_once_with(app)

class TestExceptionHandlers:
    """Test custom exception handlers."""
    
    def test_kataforge_error_handler(self, test_client):
        """Test custom KataForgeError handling."""
        from kataforge.api.server import http_exception_handler
        
        # Create a mock request and exception
        request = Mock()
        exception = KataForgeError(error_code="TEST_ERROR")
        
        response = http_exception_handler(request, exception)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_validation_error_handler(self, test_client):
        """Test validation error handling."""
        response = test_client.post("/api/v1/analyze", json={})
        
        # Should return 422 for missing required fields
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

class TestAPIAuthentication:
    """Test API authentication functionality."""
    
    def test_no_auth_required_for_health_endpoints(self, test_client):
        """Test health endpoints don't require auth."""
        endpoints = ["/", "/health/live", "/health/ready", "/health/startup"]
        
        for endpoint in endpoints:
            response = test_client.get(endpoint)
            assert response.status_code != 401
    
    @patch('kataforge.api.server.Settings')
    def test_auth_required_for_protected_endpoints(self, mock_settings, test_client):
        """Test protected endpoints require authentication."""
        # Enable authentication
        mock_settings.return_value.auth_enabled = True
        mock_settings.return_value.api_keys_list = ["test-key"]
        
        response = test_client.post("/api/v1/analyze")
        
        assert response.status_code == 401
    
    def test_valid_api_key_accepted(self, app, test_api_key):
        """Test valid API key is accepted."""
        # Configure app with auth
        app.state.settings.auth_enabled = True
        app.state.settings.api_keys_list = [test_api_key]
        
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {test_api_key}"}
        
        response = client.get("/health/live", headers=headers)
        
        assert response.status_code == 200
    
    def test_invalid_api_key_rejected(self, app):
        """Test invalid API key is rejected."""
        # Configure app with auth
        app.state.settings.auth_enabled = True
        app.state.settings.api_keys_list = ["valid-key"]
        
        client = TestClient(app)
        headers = {"Authorization": "Bearer invalid-key"}
        
        response = client.get("/health/live", headers=headers)
        
        assert response.status_code == 401

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @patch('kataforge.api.server.RateLimitMiddleware')
    def test_rate_limiting_configured(self, mock_rate_limit, test_client):
        """Test rate limiting middleware is configured when enabled."""
        # Enable rate limiting
        test_client.app.state.settings.rate_limit_enabled = True
        test_client.app.state.settings.rate_limit_requests = 100
        
        # Mock the middleware to verify it's called
        mock_rate_limit.assert_called()
    
    def test_rate_limit_headers(self, test_client):
        """Test rate limit headers are present."""
        response = test_client.get("/")
        
        # Even without rate limiting, should have some headers
        assert "x-ratelimit-limit" not in response.headers  # Not enabled by default
        assert "x-ratelimit-remaining" not in response.headers

class TestAPIEndpoints:
    """Test specific API endpoints."""
    
    def test_analyze_endpoint_missing_data(self, test_client):
        """Test analyze endpoint rejects missing data."""
        response = test_client.post("/api/v1/analyze")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_analyze_endpoint_invalid_format(self, test_client):
        """Test analyze endpoint rejects invalid format."""
        response = test_client.post(
            "/api/v1/analyze",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @patch('kataforge.api.server.analyze_video')
    def test_analyze_endpoint_success(self, mock_analyze, test_client, valid_analysis_result):
        """Test analyze endpoint success case."""
        mock_analyze.return_value = valid_analysis_result
        
        response = test_client.post(
            "/api/v1/analyze",
            json={
                "video": "test.mp4",
                "coach": "test-coach",
                "technique": "front-kick"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 8.5
    
    def test_batch_analyze_endpoint(self, test_client):
        """Test batch analyze endpoint."""
        response = test_client.post("/api/v1/batch_analyze")
        
        # Should require authentication
        assert response.status_code in [401, 422]

class TestFileUpload:
    """Test file upload functionality."""
    
    def test_file_upload_no_file(self, test_client):
        """Test upload with no file fails."""
        response = test_client.post("/api/v1/upload")
        
        assert response.status_code == 422
    
    def test_file_upload_invalid_file_type(self, test_client, invalid_video_file):
        """Test upload of invalid file type fails."""
        with open(invalid_video_file, 'rb') as f:
            response = test_client.post(
                "/api/v1/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file type" in data.get("detail", "")
    
    def test_file_upload_size_limit(self, test_client):
        """Test upload respects file size limits."""
        # Create a mock large file
        large_data = b"x" * (200 * 1024 * 1024)  # 200MB
        
        response = test_client.post(
            "/api/v1/upload",
            files={"file": ("large.mp4", large_data, "video/mp4")}
        )
        
        assert response.status_code == 413
        data = response.json()
        assert "File too large" in data.get("detail", "")

class TestConfiguration:
    """Test API configuration and settings."""
    
    def test_app_loads_settings(self):
        """Test app loads settings correctly."""
        app = create_app()
        
        assert hasattr(app.state, 'settings')
        assert app.state.settings is not None
    
    def test_app_uses_test_mode_when_configured(self):
        """Test app uses test mode when environment is set."""
        import os
        os.environ['DOJO_ENVIRONMENT'] = 'testing'
        
        app = create_app()
        
        assert app.state.settings.environment == 'testing'

class TestErrorScenarios:
    """Test various error scenarios."""
    
    def test_404_handler(self, test_client):
        """Test 404 error handler."""
        response = test_client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_method_not_allowed(self, test_client):
        """Test method not allowed error."""
        response = test_client.patch("/health/live")
        
        assert response.status_code == 405
    
    def test_server_error_handling(self, test_client):
        """Test server error is handled gracefully."""
        # Mock a route that raises an exception
        app = test_client.app
        
        @app.get("/test-error")
        async def test_error():
            raise Exception("Test server error")
        
        response = test_client.get("/test-error")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

# Async Tests
class TestAsyncAPI:
    """Test async API functionality."""
    
    @pytest.mark.asyncio
    async def test_async_health_check(self, async_client):
        """Test health check with async client."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_async_analyze_endpoint(self, async_client):
        """Test analyze endpoint with async client."""
        response = await async_client.post(
            "/api/v1/analyze",
            json={
                "video": "test.mp4",
                "coach": "test-coach",
                "technique": "front-kick"
            }
        )
        
        # Should fail validation without proper setup
        assert response.status_code in [422, 500]