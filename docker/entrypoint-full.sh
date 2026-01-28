#!/bin/bash
# =============================================================================
# Dojo Manager Full Stack Entrypoint
# =============================================================================
#
# This entrypoint manages all services for the full stack Docker image:
# - Dojo Manager API (FastAPI)
# - LLM Server (Ollama or llama.cpp depending on backend)
# - Gradio Web UI
#
# Environment Variables:
#   DOJO_API_HOST       - API server host (default: 0.0.0.0)
#   DOJO_API_PORT       - API server port (default: 8000)
#   DOJO_LLM_BACKEND    - LLM backend: "ollama" or "llamacpp" (default: ollama)
#   DOJO_OLLAMA_HOST    - Ollama server host (default: http://localhost:11434)
#   DOJO_VISION_MODEL   - Vision model for Ollama (default: llava:7b)
#   DOJO_TEXT_MODEL     - Text model for Ollama (default: mistral:7b)
#   GRADIO_SERVER_NAME  - Gradio server host (default: 0.0.0.0)
#   GRADIO_SERVER_PORT  - Gradio server port (default: 7860)
#   LLAMA_MODEL_PATH    - Path to GGUF model for llama.cpp
#   LLAMA_MMPROJ_PATH   - Path to multimodal projector for llama.cpp
#
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DOJO_API_HOST="${DOJO_API_HOST:-0.0.0.0}"
DOJO_API_PORT="${DOJO_API_PORT:-8000}"
DOJO_LLM_BACKEND="${DOJO_LLM_BACKEND:-ollama}"
DOJO_OLLAMA_HOST="${DOJO_OLLAMA_HOST:-http://localhost:11434}"
DOJO_VISION_MODEL="${DOJO_VISION_MODEL:-llava:7b}"
DOJO_TEXT_MODEL="${DOJO_TEXT_MODEL:-mistral:7b}"
GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

# llama.cpp specific
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/models/llava-v1.5-7b-q4_k.gguf}"
LLAMA_MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-/models/mmproj-model-f16.gguf}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"

# Timeouts
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-120}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-2}"
MODEL_PULL_TIMEOUT="${MODEL_PULL_TIMEOUT:-600}"

# PID tracking for cleanup
API_PID=""
LLM_PID=""
GRADIO_PID=""

# Marker file for first-run model pulling
MODELS_READY_MARKER="/data/.models_ready"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [entrypoint] $*"
}

log_info() {
    log "INFO: $*"
}

log_warn() {
    log "WARN: $*"
}

log_error() {
    log "ERROR: $*" >&2
}

# -----------------------------------------------------------------------------
# Signal Handling
# -----------------------------------------------------------------------------

cleanup() {
    log_info "Received shutdown signal, stopping services..."
    
    # Stop Gradio first (depends on others)
    if [[ -n "$GRADIO_PID" ]] && kill -0 "$GRADIO_PID" 2>/dev/null; then
        log_info "Stopping Gradio UI (PID: $GRADIO_PID)..."
        kill -TERM "$GRADIO_PID" 2>/dev/null || true
        wait "$GRADIO_PID" 2>/dev/null || true
    fi
    
    # Stop LLM server
    if [[ -n "$LLM_PID" ]] && kill -0 "$LLM_PID" 2>/dev/null; then
        log_info "Stopping LLM server (PID: $LLM_PID)..."
        kill -TERM "$LLM_PID" 2>/dev/null || true
        wait "$LLM_PID" 2>/dev/null || true
    fi
    
    # Stop API server last
    if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
        log_info "Stopping API server (PID: $API_PID)..."
        kill -TERM "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    
    log_info "All services stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# -----------------------------------------------------------------------------
# Health Check Functions
# -----------------------------------------------------------------------------

wait_for_health() {
    local name="$1"
    local url="$2"
    local timeout="${3:-$HEALTH_CHECK_TIMEOUT}"
    local interval="${4:-$HEALTH_CHECK_INTERVAL}"
    
    log_info "Waiting for $name to become healthy at $url..."
    
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_info "$name is healthy"
            return 0
        fi
        
        sleep "$interval"
        elapsed=$((elapsed + interval))
        
        if [[ $((elapsed % 10)) -eq 0 ]]; then
            log_info "Still waiting for $name... (${elapsed}s elapsed)"
        fi
    done
    
    log_error "$name failed to become healthy within ${timeout}s"
    return 1
}

# -----------------------------------------------------------------------------
# Service Start Functions
# -----------------------------------------------------------------------------

start_api_server() {
    log_info "Starting Dojo Manager API server on ${DOJO_API_HOST}:${DOJO_API_PORT}..."
    
    python -m dojo_manager.api.server &
    API_PID=$!
    
    log_info "API server started (PID: $API_PID)"
    
    # Wait for health
    if ! wait_for_health "API server" "http://localhost:${DOJO_API_PORT}/health/live"; then
        log_error "API server failed to start"
        return 1
    fi
}

start_ollama() {
    log_info "Starting Ollama server..."
    
    # Set Ollama host to bind to all interfaces
    export OLLAMA_HOST="0.0.0.0"
    
    ollama serve &
    LLM_PID=$!
    
    log_info "Ollama server started (PID: $LLM_PID)"
    
    # Wait for health
    if ! wait_for_health "Ollama" "http://localhost:11434/"; then
        log_error "Ollama failed to start"
        return 1
    fi
    
    # Pull models on first run
    pull_ollama_models
}

pull_ollama_models() {
    if [[ -f "$MODELS_READY_MARKER" ]]; then
        log_info "Models already pulled (marker exists)"
        return 0
    fi
    
    log_info "First run detected, pulling required models..."
    
    # Pull vision model
    log_info "Pulling vision model: $DOJO_VISION_MODEL (this may take a while)..."
    if timeout "$MODEL_PULL_TIMEOUT" ollama pull "$DOJO_VISION_MODEL"; then
        log_info "Vision model pulled successfully"
    else
        log_warn "Failed to pull vision model, continuing anyway..."
    fi
    
    # Pull text model
    log_info "Pulling text model: $DOJO_TEXT_MODEL..."
    if timeout "$MODEL_PULL_TIMEOUT" ollama pull "$DOJO_TEXT_MODEL"; then
        log_info "Text model pulled successfully"
    else
        log_warn "Failed to pull text model, continuing anyway..."
    fi
    
    # Create marker
    mkdir -p "$(dirname "$MODELS_READY_MARKER")"
    touch "$MODELS_READY_MARKER"
    log_info "Models ready marker created"
}

start_llama_cpp() {
    log_info "Starting llama.cpp server with Vulkan backend..."
    
    # Verify model files exist
    if [[ ! -f "$LLAMA_MODEL_PATH" ]]; then
        log_error "Model file not found: $LLAMA_MODEL_PATH"
        return 1
    fi
    
    if [[ ! -f "$LLAMA_MMPROJ_PATH" ]]; then
        log_warn "Multimodal projector not found: $LLAMA_MMPROJ_PATH"
        log_warn "Vision capabilities may be limited"
    fi
    
    # Start llama.cpp server
    # The server binary is expected to be in PATH (from Nix derivation)
    llama-server \
        --model "$LLAMA_MODEL_PATH" \
        --mmproj "$LLAMA_MMPROJ_PATH" \
        --host "$LLAMA_HOST" \
        --port "$LLAMA_PORT" \
        --n-gpu-layers "$LLAMA_N_GPU_LAYERS" \
        --ctx-size 4096 \
        --threads 4 &
    LLM_PID=$!
    
    log_info "llama.cpp server started (PID: $LLM_PID)"
    
    # Wait for health
    if ! wait_for_health "llama.cpp" "http://localhost:${LLAMA_PORT}/health"; then
        log_error "llama.cpp failed to start"
        return 1
    fi
}

start_llm_server() {
    case "$DOJO_LLM_BACKEND" in
        ollama)
            start_ollama
            ;;
        llamacpp)
            start_llama_cpp
            ;;
        *)
            log_error "Unknown LLM backend: $DOJO_LLM_BACKEND"
            return 1
            ;;
    esac
}

start_gradio() {
    log_info "Starting Gradio UI on ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}..."
    
    # Determine LLM URL based on backend
    local llm_url
    case "$DOJO_LLM_BACKEND" in
        ollama)
            llm_url="http://localhost:11434"
            ;;
        llamacpp)
            llm_url="http://localhost:${LLAMA_PORT}"
            ;;
    esac
    
    # Start Gradio UI
    python -m dojo_manager.cli.main ui \
        --host "$GRADIO_SERVER_NAME" \
        --port "$GRADIO_SERVER_PORT" \
        --api-url "http://localhost:${DOJO_API_PORT}" \
        --ollama-url "$llm_url" &
    GRADIO_PID=$!
    
    log_info "Gradio UI started (PID: $GRADIO_PID)"
    
    # Wait for health
    if ! wait_for_health "Gradio UI" "http://localhost:${GRADIO_SERVER_PORT}/"; then
        log_error "Gradio UI failed to start"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    log_info "========================================="
    log_info "Dojo Manager Full Stack Container"
    log_info "========================================="
    log_info "Backend: $DOJO_LLM_BACKEND"
    log_info "API Port: $DOJO_API_PORT"
    log_info "Gradio Port: $GRADIO_SERVER_PORT"
    log_info "========================================="
    
    # Create data directory
    mkdir -p /data
    
    # Start services in order
    log_info "Starting services..."
    
    # 1. Start API server
    if ! start_api_server; then
        log_error "Failed to start API server, exiting"
        exit 1
    fi
    
    # 2. Start LLM server
    if ! start_llm_server; then
        log_error "Failed to start LLM server, exiting"
        cleanup
        exit 1
    fi
    
    # 3. Start Gradio UI
    if ! start_gradio; then
        log_error "Failed to start Gradio UI, exiting"
        cleanup
        exit 1
    fi
    
    log_info "========================================="
    log_info "All services started successfully!"
    log_info "========================================="
    log_info "API:    http://localhost:${DOJO_API_PORT}"
    log_info "Gradio: http://localhost:${GRADIO_SERVER_PORT}"
    log_info "========================================="
    
    # Wait for any child process to exit
    # This keeps the container running and handles signals
    wait -n || true
    
    # If we get here, a service crashed
    log_error "A service exited unexpectedly"
    cleanup
    exit 1
}

main "$@"
