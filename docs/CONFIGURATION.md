# KataForge Configuration System

## Overview

KataForge uses a comprehensive configuration system with multiple layers of configuration:

1. **Environment Variables** (highest priority) - Prefixed with `DOJO_`
2. **Configuration Files** - YAML-based configuration
3. **Default Values** (lowest priority) - Sensible defaults for all settings

## Configuration Files

### Main Configuration Files

- **`config/base.yaml`**: Base configuration with default settings
- **`config/framework16.yaml`**: Framework 16 laptop optimization (175W)
- **`config/framework16_production.yaml`**: Production configuration for Framework 16

### Profile Configuration

- **`config/profiles/rocm.yaml`**: AMD ROCm-specific configuration
- Additional profiles can be created for specific hardware configurations

### Configuration Loading Priority

1. Environment variables (`DOJO_*`)
2. Profile-specific configuration (if `DOJO_PROFILE` is set)
3. Base configuration (`config/base.yaml`)
4. Default values from `Settings` class

## Environment Variables

All settings can be overridden using environment variables prefixed with `DOJO_`.

### Common Environment Variables

```bash
# Core settings
export DOJO_ENVIRONMENT=development  # development, staging, production, testing
export DOJO_DEBUG=true              # Enable debug mode
export DOJO_DATA_DIR=~/.kataforge/data  # Data directory

# API Server
export DOJO_API_HOST=0.0.0.0        # API host
export DOJO_API_PORT=8000           # API port
export DOJO_API_WORKERS=4          # Worker processes
export DOJO_API_RELOAD=true         # Auto-reload in development

# GPU Configuration
export DOJO_MODEL_DEVICE=auto       # auto, cpu, cuda, rocm
export DOJO_GPU_MEMORY_FRACTION=0.9 # GPU memory usage

# LLM Configuration
export DOJO_LLM_BACKEND=ollama      # ollama or llamacpp
export DOJO_OLLAMA_HOST=http://localhost:11434
export DOJO_OLLAMA_VISION_MODEL=llava:7b
export DOJO_OLLAMA_TEXT_MODEL=mistral:7b

# Voice System
export DOJO_TTS_ENABLED=true        # Enable text-to-speech
export DOJO_TTS_PROVIDER=piper      # piper, coqui, edge, browser
export DOJO_STT_ENABLED=true        # Enable speech-to-text
export DOJO_STT_PROVIDER=whisper    # whisper, browser

# Security (production)
export DOJO_AUTH_ENABLED=true       # Enable authentication
export DOJO_JWT_SECRET_KEY=your_strong_secret_key_here
```

### GPU-Specific Environment Variables

#### ROCm (AMD)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export AMD_LOG_LEVEL=3
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
```

#### CUDA (NVIDIA)
```bash
export CUDA_VISIBLE_DEVICES=0
```

#### Vulkan (Intel/AMD)
```bash
export ENABLE_VULKAN_COMPUTE=1
export VULKAN_SDK=/usr/local/vulkan
```

## Settings Reference

### Application Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `app_name` | string | "kataforge" | Application name |
| `environment` | enum | "development" | Environment (development, staging, production, testing) |
| `debug` | bool | false | Enable debug mode |

### API Server Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `api_host` | string | "0.0.0.0" | API server host |
| `api_port` | int | 8000 | API server port |
| `api_workers` | int | 1 | Number of worker processes |
| `api_reload` | bool | false | Auto-reload on code changes |
| `request_timeout` | int | 30 | Request timeout in seconds |
| `max_request_size` | int | 104857600 | Max request body size in bytes |
| `shutdown_timeout` | int | 30 | Graceful shutdown timeout |

### Model & Inference Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_path` | string | null | Path to trained model file |
| `model_dir` | string | "~/.kataforge/models" | Directory for model files |
| `model_device` | string | "auto" | Device for inference (auto, cpu, cuda, rocm) |
| `gpu_memory_fraction` | float | 0.9 | Fraction of GPU memory to use |
| `rocm_arch` | string | null | ROCm GPU architecture (e.g., gfx1100) |

### Data & Storage Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `data_dir` | string | "~/.kataforge/data" | Base data directory |

### Logging Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `log_level` | enum | "INFO" | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `log_format` | enum | "auto" | Log format (json, console, auto) |
| `log_file` | string | null | Path to log file |

### Security Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `auth_enabled` | bool | false | Enable API authentication |
| `api_keys` | string | "" | Comma-separated API keys |
| `api_keys_file` | string | null | File containing API keys |
| `jwt_secret_key` | string | "CHANGE-ME-IN-PRODUCTION" | JWT secret key |
| `jwt_algorithm` | string | "HS256" | JWT signing algorithm |
| `jwt_expiry_hours` | int | 24 | JWT token expiry hours |
| `tls_enabled` | bool | false | Enable TLS/SSL |
| `tls_cert_file` | string | null | TLS certificate file |
| `tls_key_file` | string | null | TLS private key file |

### Rate Limiting Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `rate_limit_enabled` | bool | true | Enable rate limiting |
| `rate_limit_requests` | int | 100 | Max requests per window |
| `rate_limit_window` | int | 60 | Rate limit window in seconds |
| `rate_limit_burst` | int | 20 | Burst allowance |

### CORS Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `cors_origins` | string | "" | Comma-separated allowed origins |
| `cors_allow_credentials` | bool | false | Allow credentials in CORS |

### Gradio UI Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `gradio_host` | string | "0.0.0.0" | Gradio server host |
| `gradio_port` | int | 7860 | Gradio server port |
| `gradio_share` | bool | false | Create public share link |

### Ollama / LLM Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ollama_host` | string | "http://localhost:11434" | Ollama API URL |
| `ollama_vision_model` | string | "llava:7b" | Vision model for analysis |
| `ollama_text_model` | string | "mistral:7b" | Text model for feedback |
| `ollama_timeout` | int | 120 | Request timeout in seconds |
| `llm_backend` | string | "ollama" | LLM backend (ollama, llamacpp) |

### Voice System Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tts_enabled` | bool | true | Enable text-to-speech |
| `tts_provider` | string | "piper" | TTS provider (piper, coqui, edge, browser) |
| `tts_voice` | string | "en_US-lessac-medium" | TTS voice model |
| `tts_speed` | float | 1.0 | Speech rate (0.5-2.0) |
| `stt_enabled` | bool | true | Enable speech-to-text |
| `stt_provider` | string | "whisper" | STT provider (whisper, browser) |
| `stt_model` | string | "base" | Whisper model size |
| `stt_language` | string | "en" | Recognition language |
| `voice_activation_phrase` | string | "hey coach" | Wake word for activation |
| `voice_feedback_auto_play` | bool | true | Auto-play TTS feedback |

### Feature Flags

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_metrics` | bool | true | Enable Prometheus metrics |
| `enable_tracing` | bool | false | Enable OpenTelemetry tracing |

## Profile Management

### Using Profiles

Set the `DOJO_PROFILE` environment variable to use a specific hardware profile:

```bash
# Use ROCm profile
export DOJO_PROFILE=rocm
kataforge train --coach=nagato --technique=roundhouse

# Or use for a single command
DOJO_PROFILE=cuda kataforge analyze --video=technique.mp4 --coach=nagato
```

### Creating Custom Profiles

1. Create a new profile file in `config/profiles/`:

```bash
# Create a custom profile for your GPU
cp config/profiles/rocm.yaml config/profiles/mygpu.yaml
```

2. Edit the profile to match your hardware:

```yaml
# config/profiles/mygpu.yaml
device: "cuda"
power_limit: 200
precision: "fp16"
rocm_arch: "gfx1100"
```

3. Use your profile:

```bash
export DOJO_PROFILE=mygpu
kataforge train --coach=nagato
```

## Configuration Validation

KataForge automatically validates configuration in production environments:

- **JWT Secret**: Must be strong and ≥64 characters in production
- **Authentication**: Must be enabled in production
- **Debug Mode**: Must be disabled in production
- **TLS**: Required for public endpoints in production
- **Rate Limiting**: Must be enabled in production

## Examples

### Development Configuration

```bash
# Development setup
export DOJO_ENVIRONMENT=development
export DOJO_DEBUG=true
export DOJO_API_RELOAD=true
export DOJO_AUTH_ENABLED=false

kataforge serve
```

### Production Configuration

```bash
# Production setup
export DOJO_ENVIRONMENT=production
export DOJO_DEBUG=false
export DOJO_AUTH_ENABLED=true
export DOJO_JWT_SECRET_KEY=$(openssl rand -hex 32)
export DOJO_API_WORKERS=4
export DOJO_RATE_LIMIT_ENABLED=true

kataforge serve
```

### Framework 16 Optimization

```bash
# Framework 16 (AMD Ryzen 9 7840HS + RX 7700S)
export DOJO_PROFILE=framework16
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

kataforge train --coach=nagato --device=rocm
```

## Configuration File Reference

### base.yaml

```yaml
# Base Configuration
version: 1
app_name: "kataforge"
environment: "development"
debug: false

# Data Management
data_dir: "~/.kataforge/data"

# Core Components
logging:
  level: "INFO"
  format: "json"

# LLM Settings
llm:
  backend: "ollama"
  endpoint: "http://localhost:11434/api/generate"

# Training Defaults
training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001

# Security
security:
  jwt_expiry_hours: 24
  api_key_rotation_days: 30

# UI Configuration
ui:
  theme: "dark"
  enable_telemetry: false
```

### framework16.yaml

```yaml
# Framework 16 Configuration (175W)
version: 1
extends: "base"

hardware:
  device: "rocm"
  gpu_memory_gb: 8
  architecture: "RDNA3"

power:
  gpu_power_limit: 175        # Watts - OPTIMIZED
  gpu_temp_target: 85         # Celsius

training:
  batch_size: 4
  sequence_length: 32
  mixed_precision: true       # FP16 for 2x speed
  gradient_checkpointing: true
  gradient_accumulation_steps: 4
  num_workers: 4
  pin_memory: true
  learning_rate: 0.0001
  epochs: 100
```

## Configuration Best Practices

1. **Use Environment Variables** for sensitive data (API keys, JWT secrets)
2. **Use Profiles** for hardware-specific configurations
3. **Validate Configuration** before production deployment
4. **Document Custom Profiles** for team members
5. **Test Configuration Changes** in staging before production

## Troubleshooting

### Configuration Not Loading

```bash
# Check environment variables
env | grep DOJO_

# Check current settings
kataforge status

# Validate configuration
python -c "from kataforge.core.settings import get_settings; print(get_settings().model_dump_json(indent=2))"
```

### Invalid Configuration

```bash
# Check for validation errors
kataforge status 2>&1 | grep -i error

# Fix issues and retry
```

### Missing Configuration Files

```bash
# Initialize configuration
kataforge init

# Or manually create config directory
mkdir -p config/profiles
cp config/framework16.yaml config/profiles/rocm.yaml
```

## Advanced Configuration

### Custom Model Paths

```bash
# Use custom model directory
export DOJO_MODEL_DIR=/path/to/custom/models
kataforge analyze --model=/path/to/custom/model.pt
```

### Multiple GPU Configuration

```bash
# For multi-GPU systems
export CUDA_VISIBLE_DEVICES=0,1
export DOJO_MODEL_DEVICE=cuda
kataforge train --batch-size=8 --workers=2
```

### Custom Voice Models

```bash
# Use custom TTS voice
export DOJO_TTS_VOICE=en_GB-vctk-medium
kataforge ui --tts-provider=piper
```
