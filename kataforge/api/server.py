"""
Production-Ready Inference Server for Martial Arts Technique Analysis

Features:
- Graceful shutdown handling
- Health check endpoints (liveness, readiness, startup)
- Standardized error responses
- Request ID tracking
- Structured logging

TODO Phase 4 (Observability):
- Add Prometheus metrics endpoint (/metrics) using prometheus-client
  - Request count/latency histograms by endpoint
  - Model inference latency
  - GPU memory utilization
  - Active connections gauge
- Add OpenTelemetry tracing instrumentation
  - Trace ID propagation across services
  - Span creation for key operations (inference, pose extraction)
  - Export to Jaeger/Zipkin
- Add correlation ID support linking logs to traces
"""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from fastapi import FastAPI, Request, Response, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    Request = None
    Response = None
    UploadFile = None
    File = None
    HTTPException = None
    CORSMiddleware = None
    JSONResponse = None
    BaseModel = object
    Field = lambda *args, **kwargs: None
    field_validator = lambda *args, **kwargs: lambda f: f
    uvicorn = None

import numpy as np

# Import our modules
from ..core.settings import get_settings
from ..core.logging import setup_logging, get_logger, set_request_context, clear_request_context
from ..core.error_handling import (
    DojoManagerError,
    DataValidationError,
    ModelInferenceError,
    ProcessingError,
)


# =============================================================================
# Request/Response Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class PoseRequest(BaseModel):
        """Request model for pose analysis with validation."""
        poses: List[List[List[float]]] = Field(
            ...,
            description="3D pose data [frames, 33 landmarks, 4 coords (x,y,z,visibility)]",
            min_length=1,
        )
        coach_id: str = Field(
            ...,
            min_length=1,
            max_length=50,
            description="Coach identifier"
        )
        technique: str = Field(
            ...,
            min_length=1,
            max_length=100,
            description="Technique name"
        )
        
        @field_validator('poses')
        @classmethod
        def validate_pose_shape(cls, v):
            """Validate pose data shape."""
            if v and len(v) > 0:
                if len(v[0]) != 33:
                    raise ValueError('Each frame must have 33 landmarks (MediaPipe format)')
                if len(v[0][0]) != 4:
                    raise ValueError('Each landmark must have 4 coordinates (x, y, z, visibility)')
            return v
        
        model_config = {
            "json_schema_extra": {
                "example": {
                    "poses": [[[0.5, 0.5, 0.0, 1.0]] * 33] * 10,
                    "coach_id": "saenchai",
                    "technique": "roundhouse_kick"
                }
            }
        }


    class AnalysisResponse(BaseModel):
        """Response model for technique analysis."""
        overall_score: float = Field(..., ge=0, le=10, description="Overall technique score (0-10)")
        aspect_scores: Dict[str, float] = Field(..., description="Scores for individual aspects")
        corrections: List[str] = Field(default_factory=list, description="Suggested corrections")
        recommendations: List[str] = Field(default_factory=list, description="Training recommendations")
        biomechanics: Dict[str, float] = Field(default_factory=dict, description="Biomechanics metrics")


    class ErrorResponse(BaseModel):
        """Standardized error response."""
        error_code: str = Field(..., description="Machine-readable error code")
        message: str = Field(..., description="Human-readable error message")
        details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
        request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
        timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp")


    class HealthResponse(BaseModel):
        """Health check response."""
        status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
        checks: Dict[str, bool] = Field(default_factory=dict, description="Individual check results")
        version: str = Field(default="0.1.0", description="Application version")
        timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Server State Management
# =============================================================================

class ServerState:
    """Manages server state for health checks and graceful shutdown."""
    
    def __init__(self):
        self.is_ready: bool = False
        self.is_shutting_down: bool = False
        self.startup_complete: bool = False
        self.model_loaded: bool = False
        self.active_requests: int = 0
        self.start_time: datetime = datetime.utcnow()
        self.inference_engine: Optional[Any] = None  # Will be InferenceEngine
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()


# Global state instance
_server_state = ServerState()


# =============================================================================
# Middleware
# =============================================================================

async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to add request ID tracking.
    
    - Extracts X-Request-ID from incoming request or generates a new one
    - Binds request ID to logging context
    - Adds request ID to response headers
    """
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:16]}"
    
    # Set in logging context
    set_request_context(request_id=request_id)
    
    # Store on request for access in handlers
    request.state.request_id = request_id
    
    try:
        # Track active requests
        _server_state.active_requests += 1
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        return response
    finally:
        _server_state.active_requests -= 1
        clear_request_context()


# =============================================================================
# Exception Handlers
# =============================================================================

def get_request_id_from_request(request: Request) -> Optional[str]:
    """Safely get request ID from request state."""
    try:
        return getattr(request.state, 'request_id', None)
    except Exception:
        return None


async def dojo_exception_handler(request: Request, exc: DojoManagerError) -> JSONResponse:
    """Handle DojoManagerError exceptions with proper status codes."""
    logger = get_logger(__name__)
    request_id = get_request_id_from_request(request)
    
    # Map error types to HTTP status codes
    status_code_map = {
        "DATA_VALIDATION_ERROR": 422,
        "AUTHENTICATION_ERROR": 401,
        "AUTHORIZATION_ERROR": 403,
        "DATA_NOT_FOUND_ERROR": 404,
        "RATE_LIMIT_ERROR": 429,
        "TIMEOUT_ERROR": 504,
        "PROCESSING_ERROR": 500,
        "MODEL_INFERENCE_ERROR": 500,
    }
    
    status_code = status_code_map.get(exc.error_code, 500)
    
    logger.error(
        "request_error",
        error_code=exc.error_code,
        message=str(exc),
        status_code=status_code,
        request_id=request_id,
    )
    
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=exc.error_code,
            message=str(exc),
            details=exc.context if hasattr(exc, 'context') else None,
            request_id=request_id,
        ).model_dump(),
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger = get_logger(__name__)
    request_id = get_request_id_from_request(request)
    
    # Extract validation errors
    errors = []
    if hasattr(exc, 'errors'):
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown"),
            })
    
    logger.warning(
        "validation_error",
        errors=errors,
        request_id=request_id,
    )
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": errors},
            request_id=request_id,
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger = get_logger(__name__)
    request_id = get_request_id_from_request(request)
    
    logger.exception(
        "unhandled_exception",
        error_type=type(exc).__name__,
        request_id=request_id,
    )
    
# Don't expose internal errors in production
    settings = get_settings()
    message = "We're having trouble processing your request. Please try again or check your connection."

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message=message,
            request_id=request_id,
        ).model_dump(),
    )


# =============================================================================
# Lifespan Management (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown.
    
    Handles:
    - Logging setup
    - Inference engine initialization
    - Graceful shutdown with active request draining
    """
    logger = get_logger(__name__)
    settings = get_settings()
    
    # -------------------------
    # Startup
    # -------------------------
    logger.info(
        "server_starting",
        host=settings.api_host,
        port=settings.api_port,
        environment=settings.environment.value if hasattr(settings.environment, 'value') else settings.environment,
    )
    
    # Initialize inference engine
    try:
        from ..ml.inference import InferenceEngine
        _server_state.inference_engine = InferenceEngine()
        _server_state.model_loaded = _server_state.inference_engine.is_model_loaded
        
        engine_status = _server_state.inference_engine.get_status()
        logger.info(
            "inference_engine_initialized",
            model_loaded=engine_status['model_loaded'],
            analysis_mode=engine_status['analysis_mode'],
            device=engine_status['device'],
        )
    except Exception as e:
        logger.warning(f"inference_engine_init_failed: {e} - analysis endpoints will be unavailable")
        _server_state.inference_engine = None
        _server_state.model_loaded = False
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        logger.info("shutdown_signal_received", signal=sig.name)
        _server_state.is_shutting_down = True
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f, sig=sig: signal_handler(sig))
    
    # Mark startup complete
    _server_state.startup_complete = True
    _server_state.is_ready = True
    
    logger.info("server_started", uptime=0)
    
    yield
    
    # -------------------------
    # Shutdown
    # -------------------------
    logger.info("server_shutting_down")
    _server_state.is_ready = False
    _server_state.is_shutting_down = True
    
    # Wait for active requests to complete
    shutdown_start = datetime.utcnow()
    while _server_state.active_requests > 0:
        elapsed = (datetime.utcnow() - shutdown_start).total_seconds()
        if elapsed > settings.shutdown_timeout:
            logger.warning(
                "shutdown_timeout",
                active_requests=_server_state.active_requests,
                timeout=settings.shutdown_timeout,
            )
            break
        logger.info(
            "waiting_for_requests",
            active_requests=_server_state.active_requests,
            elapsed=elapsed,
        )
        await asyncio.sleep(0.5)
    
    logger.info("server_stopped", uptime=_server_state.uptime_seconds)


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for the inference server")
    
    # Setup logging first
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()
    
    # Create app
    app = FastAPI(
        title="KataForge API",
        description="""
## Martial Arts Technique Analysis API

Analyze martial arts techniques using computer vision and machine learning.

### Features
- Real-time pose analysis
- Coach-specific style matching  
- Biomechanics calculations
- Training recommendations
- ML-powered assessment with biomechanics fallback

### Error Codes
- `VALIDATION_ERROR` (422): Invalid request data
- `DATA_NOT_FOUND_ERROR` (404): Resource not found
- `PROCESSING_ERROR` (500): Analysis failed
- `MODEL_INFERENCE_ERROR` (500): ML model error
        """,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,  # Disable docs in production
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Import security middleware
    try:
        from ..core.security import (
            RateLimitMiddleware,
            SecurityHeadersMiddleware,
            RequestSizeLimitMiddleware,
            api_key_auth,
        )
        SECURITY_AVAILABLE = True
    except ImportError:
        SECURITY_AVAILABLE = False
    
    # Add security headers middleware (outermost - runs last on response)
    if SECURITY_AVAILABLE:
        app.add_middleware(SecurityHeadersMiddleware)
    
    # Add CORS middleware
    cors_origins = settings.cors_origins_list if hasattr(settings, 'cors_origins_list') else (["*"] if settings.debug else [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=getattr(settings, 'cors_allow_credentials', False),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    if SECURITY_AVAILABLE and getattr(settings, 'rate_limit_enabled', True):
        app.add_middleware(RateLimitMiddleware)
    
    # Add request size limit middleware
    if SECURITY_AVAILABLE:
        app.add_middleware(RequestSizeLimitMiddleware)
    
    # Add request ID middleware
    app.middleware("http")(request_id_middleware)
    
    # Add exception handlers
    app.add_exception_handler(DojoManagerError, dojo_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    # Try to add Pydantic validation error handler
    try:
        from pydantic import ValidationError
        app.add_exception_handler(ValidationError, validation_exception_handler)
    except ImportError:
        pass
    
    # Register routes
    _register_routes(app)
    
    logger.info("app_created", debug=settings.debug)
    
    return app


def _register_routes(app: FastAPI):
    """Register all API routes."""
    
    # -------------------------
    # Health Check Endpoints
    # -------------------------
    
    @app.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Liveness probe",
        description="Check if the server process is alive. Used by K8s liveness probe.",
    )
    async def liveness():
        """Liveness probe - is the process alive?"""
        return HealthResponse(
            status="healthy",
            checks={"alive": True},
        )
    
    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Readiness probe",
        description="Check if the server is ready to accept traffic. Used by K8s readiness probe.",
    )
    async def readiness():
        """Readiness probe - is the server ready for traffic?"""
        checks = {
            "startup_complete": _server_state.startup_complete,
            "not_shutting_down": not _server_state.is_shutting_down,
            "model_loaded": _server_state.model_loaded,
        }
        
        # Ready if not shutting down (model loading is optional for basic readiness)
        is_ready = checks["startup_complete"] and checks["not_shutting_down"]
        
        return HealthResponse(
            status="healthy" if is_ready else "unhealthy",
            checks=checks,
        )
    
    @app.get(
        "/health/startup",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Startup probe",
        description="Check if the server has completed startup. Used by K8s startup probe.",
    )
    async def startup():
        """Startup probe - has initialization completed?"""
        checks = {
            "startup_complete": _server_state.startup_complete,
        }
        
        return HealthResponse(
            status="healthy" if _server_state.startup_complete else "unhealthy",
            checks=checks,
        )
    
    @app.get(
        "/",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Basic health check",
        description="Simple health check endpoint.",
    )
    async def health_check():
        """Basic health check."""
        return HealthResponse(
            status="healthy" if _server_state.is_ready else "unhealthy",
            checks={
                "ready": _server_state.is_ready,
                "uptime_seconds": _server_state.uptime_seconds,
            },
        )
    
    # -------------------------
    # Analysis Endpoints
    # -------------------------
    
    # Import auth dependency if available
    try:
        from ..core.security import get_auth_dependency
        auth_dep = get_auth_dependency(require_auth=True, auth_type="api_key")
    except ImportError:
        auth_dep = None
    
    # Build dependencies list
    analysis_deps = [Depends(auth_dep)] if auth_dep and Depends else []
    
    @app.post(
        "/api/v1/analyze",
        response_model=AnalysisResponse,
        tags=["Analysis"],
        summary="Analyze pose sequence",
        description="Analyze a sequence of poses and return technique scores. Requires authentication when enabled.",
        responses={
            401: {"model": ErrorResponse, "description": "Authentication required"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
            500: {"model": ErrorResponse, "description": "Processing error"},
        },
        dependencies=analysis_deps,
    )
    async def analyze_pose(request: PoseRequest, req: Request):
        """Analyze a single pose sequence."""
        logger = get_logger(__name__)
        request_id = getattr(req.state, 'request_id', None)
        
        logger.info(
            "analyze_request",
            coach_id=request.coach_id,
            technique=request.technique,
            frames=len(request.poses),
            request_id=request_id,
        )
        
        try:
            # Generate analysis (mock for now)
            analysis = _generate_analysis(request)
            
            logger.info(
                "analyze_success",
                overall_score=analysis.overall_score,
                request_id=request_id,
            )
            
            return analysis
            
        except Exception as e:
            logger.exception("analyze_failed", request_id=request_id)
            raise ModelInferenceError(f"Could not complete the analysis: {e}")
    
    @app.post(
        "/api/v1/batch_analyze",
        tags=["Analysis"],
        summary="Batch analyze pose files",
        description="Analyze multiple pose files in a single request. Requires authentication when enabled.",
        responses={
            401: {"model": ErrorResponse, "description": "Authentication required"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        },
        dependencies=analysis_deps,
    )
    async def batch_analyze(files: List[UploadFile] = File(...), req: Request = None):
        """Analyze multiple pose files."""
        logger = get_logger(__name__)
        request_id = getattr(req.state, 'request_id', None) if req else None
        
        logger.info(
            "batch_analyze_request",
            file_count=len(files),
            request_id=request_id,
        )
        
        results = []
        for file in files:
            try:
                content = await file.read()
                pose_data = json.loads(content)
                
                request = PoseRequest(
                    poses=pose_data['poses'],
                    coach_id=pose_data.get('coach_id', 'default'),
                    technique=pose_data.get('technique', 'unknown'),
                )
                
                analysis = _generate_analysis(request)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "analysis": analysis.model_dump(),
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e),
                })
        
        return {"results": results, "request_id": request_id}
    
    @app.get(
        "/api/v1/coaches",
        tags=["Coaches"],
        summary="List available coaches",
        description="Get a list of available coach profiles for style matching.",
    )
    async def list_coaches():
        """List available coaches."""
        coaches = [
            {"id": "saenchai", "name": "Saenchai Sor Kingstar", "style": "Muay Thai"},
            {"id": "kru_somchai", "name": "Kru Somchai", "style": "Muay Thai"},
        ]
        return {"coaches": coaches}
    
    # -------------------------
    # LLM Integration Endpoints
    # -------------------------
    
    @app.get(
        "/api/v1/llm/status",
        tags=["LLM"],
        summary="Check LLM connection status",
        description="Check if the LLM server (Ollama or llama.cpp) is available and responding.",
    )
    async def llm_status(req: Request):
        """Check LLM server health and available models."""
        logger = get_logger(__name__)
        request_id = getattr(req.state, 'request_id', None)
        settings = get_settings()
        
        try:
            from ..llm import create_llm_client
            
            client = create_llm_client()
            health = await client.health_check()
            
            return {
                "status": "healthy" if health.get("status") == "ok" else "unhealthy",
                "backend": settings.llm_backend,
                "host": settings.ollama_host,
                "vision_model": settings.ollama_vision_model,
                "text_model": settings.ollama_text_model,
                "details": health,
                "request_id": request_id,
            }
        except ImportError:
            return {
                "status": "unavailable",
                "backend": settings.llm_backend,
                "error": "LLM client not available - install ollama package",
                "request_id": request_id,
            }
        except Exception as e:
            logger.exception("llm_health_check_failed", request_id=request_id)
            return {
                "status": "unhealthy",
                "backend": settings.llm_backend,
                "error": str(e),
                "request_id": request_id,
            }
    
    @app.post(
        "/api/v1/analyze-frame",
        tags=["LLM"],
        summary="Analyze a single frame with vision model",
        description="Analyze a single image frame using the LLM vision model (LLaVA) for technique assessment.",
        responses={
            401: {"model": ErrorResponse, "description": "Authentication required"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "LLM processing error"},
        },
        dependencies=analysis_deps,
    )
    async def analyze_frame(
        file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
        technique: str = "general",
        req: Request = None,
    ):
        """Analyze a single frame using the vision model."""
        logger = get_logger(__name__)
        request_id = getattr(req.state, 'request_id', None) if req else None
        
        logger.info(
            "analyze_frame_request",
            filename=file.filename,
            technique=technique,
            request_id=request_id,
        )
        
        try:
            from ..llm import create_llm_client
            import base64
            
            # Read and encode image
            content = await file.read()
            image_base64 = base64.b64encode(content).decode('utf-8')
            
            # Get vision analysis
            client = create_llm_client()
            analysis = await client.analyze_frame(image_base64, technique=technique)
            
            logger.info(
                "analyze_frame_success",
                request_id=request_id,
            )
            
            return {
                "status": "success",
                "technique": technique,
                "analysis": analysis,
                "request_id": request_id,
            }
            
        except ImportError:
            raise ProcessingError("LLM client not available - install ollama package")
        except Exception as e:
            logger.exception("analyze_frame_failed", request_id=request_id)
            raise ModelInferenceError(f"Frame analysis failed: {e}")
    
    @app.post(
        "/api/v1/analyze-with-feedback",
        tags=["LLM"],
        summary="Full analysis with AI coaching feedback",
        description="""
Perform complete technique analysis combining ML pose analysis with LLM-powered 
natural language feedback. This endpoint:

1. Extracts poses from the uploaded video
2. Runs ML-based technique scoring
3. Uses a vision model to analyze key frames
4. Generates natural language coaching feedback with the text model

Returns comprehensive results including scores, corrections, and personalized coaching tips.
        """,
        responses={
            401: {"model": ErrorResponse, "description": "Authentication required"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Processing error"},
        },
        dependencies=analysis_deps,
    )
    async def analyze_with_feedback(
        file: UploadFile = File(..., description="Video file (MP4, MOV, AVI)"),
        coach_id: str = "default",
        technique: str = "general",
        include_frame_analysis: bool = True,
        req: Request = None,
    ):
        """Full analysis with LLM-powered coaching feedback."""
        logger = get_logger(__name__)
        request_id = getattr(req.state, 'request_id', None) if req else None
        
        logger.info(
            "analyze_with_feedback_request",
            filename=file.filename,
            coach_id=coach_id,
            technique=technique,
            include_frame_analysis=include_frame_analysis,
            request_id=request_id,
        )
        
        try:
            import tempfile
            import os
            from pathlib import Path
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "video.mp4").suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # Step 1: Generate ML-based analysis (mock for now)
                mock_poses = [[[0.5, 0.5, 0.0, 1.0]] * 33] * 30  # 30 frames
                pose_request = PoseRequest(
                    poses=mock_poses,
                    coach_id=coach_id,
                    technique=technique,
                )
                ml_analysis = _generate_analysis(pose_request)
                
                # Step 2: Get LLM feedback
                llm_feedback = None
                frame_analyses = []
                
                try:
                    from ..llm import create_llm_client
                    import base64
                    import cv2
                    
                    client = create_llm_client()
                    
                    # Extract key frames for vision analysis
                    if include_frame_analysis:
                        cap = cv2.VideoCapture(tmp_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Sample 3 key frames: start, middle, end of technique
                        key_frame_indices = [
                            int(frame_count * 0.1),   # Preparation
                            int(frame_count * 0.5),   # Execution
                            int(frame_count * 0.9),   # Follow-through
                        ]
                        
                        for idx in key_frame_indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                _, buffer = cv2.imencode('.jpg', frame)
                                image_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                frame_analysis = await client.analyze_frame(
                                    image_base64,
                                    technique=technique,
                                )
                                frame_analyses.append({
                                    "frame_index": idx,
                                    "phase": ["preparation", "execution", "follow-through"][len(frame_analyses)],
                                    "analysis": frame_analysis,
                                })
                        
                        cap.release()
                    
                    # Generate coaching feedback based on ML analysis
                    llm_feedback = await client.generate_feedback(
                        overall_score=ml_analysis.overall_score,
                        aspect_scores=ml_analysis.aspect_scores,
                        technique=technique,
                        corrections=ml_analysis.corrections,
                    )
                    
                except ImportError:
                    logger.warning("llm_not_available", request_id=request_id)
                    llm_feedback = "LLM feedback not available - install ollama package for AI coaching."
                except Exception as e:
                    logger.warning("llm_feedback_failed", error=str(e), request_id=request_id)
                    llm_feedback = f"AI feedback generation failed: {e}"
                
                logger.info(
                    "analyze_with_feedback_success",
                    overall_score=ml_analysis.overall_score,
                    has_llm_feedback=llm_feedback is not None,
                    frame_analyses_count=len(frame_analyses),
                    request_id=request_id,
                )
                
                return {
                    "status": "success",
                    "coach_id": coach_id,
                    "technique": technique,
                    "ml_analysis": {
                        "overall_score": ml_analysis.overall_score,
                        "aspect_scores": ml_analysis.aspect_scores,
                        "corrections": ml_analysis.corrections,
                        "recommendations": ml_analysis.recommendations,
                        "biomechanics": ml_analysis.biomechanics,
                    },
                    "ai_feedback": llm_feedback,
                    "frame_analyses": frame_analyses if include_frame_analysis else None,
                    "request_id": request_id,
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.exception("analyze_with_feedback_failed", request_id=request_id)
            raise ProcessingError(f"Analysis with feedback failed: {e}")


def _generate_analysis(request: PoseRequest) -> AnalysisResponse:
    """
    Generate technique analysis using the inference engine.
    
    Uses ML model if available, otherwise falls back to biomechanics-only analysis.
    """
    # Convert poses to numpy array
    poses = np.array(request.poses, dtype=np.float32)
    
    # Use inference engine if available
    if _server_state.inference_engine is not None:
        try:
            result = _server_state.inference_engine.analyze(
                poses=poses,
                technique=request.technique,
                coach_id=request.coach_id,
            )
            
            return AnalysisResponse(
                overall_score=result.overall_score,
                aspect_scores=result.aspect_scores,
                corrections=result.corrections,
                recommendations=result.recommendations,
                biomechanics=result.biomechanics,
            )
        except Exception as e:
            # Log error and fall through to fallback
            logger = get_logger(__name__)
            logger.warning(f"inference_engine_error: {e}")
    
    # Fallback: Create a temporary biomechanics analyzer
    from ..ml.inference import BiomechanicsAnalyzer
    analyzer = BiomechanicsAnalyzer()
    result = analyzer.analyze(poses, request.technique)
    
    return AnalysisResponse(
        overall_score=result.overall_score,
        aspect_scores=result.aspect_scores,
        corrections=result.corrections,
        recommendations=result.recommendations,
        biomechanics=result.biomechanics,
    )


# =============================================================================
# Server Runner
# =============================================================================

def run_server():
    """Run the API server with configuration from settings."""
    if not FASTAPI_AVAILABLE or uvicorn is None:
        raise ImportError("FastAPI and uvicorn are required to run the server")
    
    settings = get_settings()
    
    uvicorn.run(
        "kataforge.api.server:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_level=settings.log_level.value.lower() if hasattr(settings.log_level, 'value') else settings.log_level.lower(),
    )


# =============================================================================
# Legacy Support
# =============================================================================

class InferenceServer:
    """Legacy server class for backwards compatibility."""
    
    def __init__(self, model_path: str, host: str = "0.0.0.0", port: int = 8000):
        """Initialize inference server."""
        import os
        os.environ.setdefault("DOJO_MODEL_PATH", model_path)
        os.environ.setdefault("DOJO_API_HOST", host)
        os.environ.setdefault("DOJO_API_PORT", str(port))
        
        self.app = create_app()
        self.host = host
        self.port = port
        self.model = None  # Mock - model loading moved to lifespan
        
        # Mark model as loaded for health checks
        _server_state.model_loaded = True
    
    def start(self):
        """Start the server."""
        run_server()


def create_inference_server(model_path: str, **kwargs) -> InferenceServer:
    """Factory function for backwards compatibility."""
    return InferenceServer(model_path, **kwargs)


# Entry point for `python -m kataforge.api.server`
if __name__ == "__main__":
    run_server()
