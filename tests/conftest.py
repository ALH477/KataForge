"""
Global fixtures and configuration for KataForge test suite.

Provides shared fixtures for unit, integration, and e2e tests.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Generator

# Import only after checking dependencies
try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    AsyncClient = None

try:
    from kataforge.api.server import create_app
    from kataforge.core.settings import Settings, get_settings
    from kataforge.core.logging import get_logger, configure_logging
    KATAFORGE_AVAILABLE = True
except ImportError:
    KATAFORGE_AVAILABLE = False

# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, no external deps)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires service startup)"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test (full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (skip on CI)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU hardware"
    )

def pytest_collection_modifyitems(config, items):
    """Add automatic markers and skip conditions."""
    for item in items:
        # Add markers based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Skip GPU tests if no GPU available
        if item.get_closest_marker("gpu"):
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(pytest.mark.skip(reason="GPU not available"))
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))

# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_video_path():
    """Path to a sample video file for testing."""
    # Create a mock video file path
    return Path(__file__).parent / "fixtures" / "sample_video.mp4"

@pytest.fixture
def sample_pose_data():
    """Sample pose data for testing ML models."""
    return {
        "total_frames": 100,
        "fps": 30.0,
        "poses": [
            {
                "landmarks": [[0.1, 0.2, 0.0] for _ in range(33)],
                "visibility": [1.0] * 33,
                "timestamp": i / 30.0
            }
            for i in range(100)
        ]
    }

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        app_name="kataforge-test",
        environment="testing",
        debug=True,
        api_host="127.0.0.1",
        api_port=8000,
        data_dir="/tmp/test-kataforge",
        log_level="INFO",
        auth_enabled=False,
        rate_limit_enabled=False,
        tls_enabled=False,
        model_dir="/tmp/test-models"
    )

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger

# =============================================================================
# API Testing Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def app():
    """Create FastAPI application instance for testing."""
    if not KATAFORGE_AVAILABLE:
        pytest.skip("KataForge not available")
    
    # Configure test environment
    import os
    os.environ["DOJO_ENVIRONMENT"] = "testing"
    os.environ["DOJO_DEBUG"] = "true"
    os.environ["DOJO_AUTH_ENABLED"] = "false"
    os.environ["DOJO_RATE_LIMIT_ENABLED"] = "false"
    
    # Clear settings cache
    try:
        from kataforge.core.settings import clear_settings_cache
        clear_settings_cache()
    except ImportError:
        pass
    
    # Create app
    app = create_app()
    
    # Setup test database/data directories
    app.state.test_mode = True
    app.state.data_dir = Path(tempfile.mkdtemp(prefix="kataforge-test-"))
    
    yield app
    
    # Cleanup
    import shutil
    if hasattr(app.state, "data_dir") and app.state.data_dir.exists():
        shutil.rmtree(app.state.data_dir)

@pytest.fixture
def test_client(app):
    """Create test client for FastAPI application."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")
    return TestClient(app)

@pytest.fixture
async def async_client(app):
    """Create async test client for FastAPI application."""
    if not AsyncClient:
        pytest.skip("AsyncClient not available")
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

# =============================================================================
# Mock Service Fixtures
# =============================================================================

@pytest.fixture
def mock_pose_extractor():
    """Mock MediaPipe pose extractor."""
    extractor = Mock()
    extractor.extract_from_video = Mock(return_value={
        "total_frames": 100,
        "fps": 30.0,
        "poses": [{"landmarks": [[0.1, 0.2, 0.0] * 33]} for _ in range(100)]
    })
    extractor.save_poses = Mock(return_value=True)
    return extractor

@pytest.fixture
def mock_biomechanics_calculator():
    """Mock biomechanics calculator."""
    calculator = Mock()
    calculator.calculate_all = Mock(return_value={
        "max_speed": 5.2,
        "peak_force": 1500.0,
        "kinetic_chain_efficiency": 85.0,
        "balance_score": 8.5,
        "timing_score": 7.8
    })
    return calculator

@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing."""
    model = Mock()
    model.eval = Mock()
    model.load_state_dict = Mock()
    model.__call__ = Mock(return_value={
        "overall_score": 8.2,
        "aspect_scores": [[7.5, 8.0, 8.5, 8.2, 8.1]]
    })
    return model

@pytest.fixture
def mock_data_loader():
    """Mock data loader for ML training."""
    loader = Mock()
    # Mock dataset with train/val splits
    loader.train_loader = Mock()
    loader.val_loader = Mock()
    loader.train_loader.__len__ = Mock(return_value=100)
    loader.val_loader.__len__ = Mock(return_value=20)
    return loader

# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def valid_video_file(temp_dir):
    """Create a valid video file for testing."""
    video_file = temp_dir / "test_video.mp4"
    # Create a minimal MP4 file (placeholder - in real tests would use actual video)
    video_file.write_bytes(b"fake mp4 data for testing")
    return video_file

@pytest.fixture
def invalid_video_file(temp_dir):
    """Create an invalid video file for testing."""
    video_file = temp_dir / "invalid_video.txt"
    video_file.write_text("This is not a video file")
    return video_file

@pytest.fixture
def valid_analysis_result():
    """Valid analysis result for API responses."""
    return {
        "video": "/test/video.mp4",
        "coach": "test-coach",
        "technique": "front-kick",
        "overall_score": 8.5,
        "aspect_scores": {
            "speed": 8.0,
            "force": 8.5,
            "timing": 8.7,
            "balance": 8.3,
            "coordination": 9.0
        },
        "biomechanics": {
            "max_speed": 5.2,
            "peak_force": 1500.0,
            "kinetic_chain_efficiency": 85.0
        },
        "corrections": ["Slightly improve chamber height"],
        "recommendations": ["Practice slow-motion drills"]
    }

# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def test_api_key():
    """Test API key for authentication."""
    return "test-api-key-12345"

@pytest.fixture
def valid_auth_headers(test_api_key):
    """Valid authentication headers for API requests."""
    return {"Authorization": f"Bearer {test_api_key}"}

@pytest.fixture
def invalid_auth_headers():
    """Invalid authentication headers for testing."""
    return {"Authorization": "Bearer invalid-key"}

# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup common test environment variables."""
    # Disable actual logging during tests to keep output clean
    monkeypatch.setenv("DOJO_LOG_LEVEL", "WARNING")
    
    # Use temporary directories for all data
    temp_data_dir = tempfile.mkdtemp(prefix="kataforge-test-data-")
    monkeypatch.setenv("DOJO_DATA_DIR", temp_data_dir)
    
    # Disable external services
    monkeypatch.setenv("DOJO_OLLAMA_HOST", "http://localhost:99999")  # Invalid port
    
    yield
    
    # Cleanup
    import shutil
    import os
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)

# =============================================================================
# Database Fixtures (if using database)
# =============================================================================

@pytest.fixture
def mock_database():
    """Mock database connection and operations."""
    db = Mock()
    db.connect = Mock(return_value=True)
    db.disconnect = Mock(return_value=True)
    db.execute = Mock(return_value=[])
    db.commit = Mock()
    db.rollback = Mock()
    return db

# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    import psutil
    import os
    
    class Monitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.start_cpu = None
            self.end_cpu = None
        
        def start(self):
            self.start_time = time.time()
            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss
            self.start_cpu = process.cpu_percent()
        
        def stop(self):
            self.end_time = time.time()
            process = psutil.Process(os.getpid())
            self.end_memory = process.memory_info().rss
            self.end_cpu = process.cpu_percent()
        
        @property
        def duration(self):
            return self.end_time - self.start_time if self.end_time and self.start_time else None
        
        @property
        def memory_delta(self):
            return (self.end_memory - self.start_memory) if self.end_memory and self.start_memory else None
        
        @property
        def cpu_delta(self):
            return (self.end_cpu - self.start_cpu) if self.end_cpu and self.start_cpu else None
    
    return Monitor()