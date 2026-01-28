# KataForge CLI Reference

## Overview

KataForge provides a comprehensive command-line interface built with Typer and Rich, offering beautiful colored output, progress bars, and structured commands for martial arts technique analysis.

## Installation

```bash
# Install via Nix
nix develop

# Or install manually
pip install kataforge
```

## Basic Usage

```bash
# Show help
kataforge --help

# Show version
kataforge --version

# Initialize training space
kataforge init --name my_dojo --data-dir ~/.kataforge/data

# Extract poses from video
kataforge extract-pose technique.mp4 --output poses.json

# Train a model
kataforge train --coach nagato --data-dir data/poses --epochs 100

# Analyze technique
kataforge analyze --video technique.mp4 --coach nagato --technique roundhouse

# Start API server
kataforge serve --host 0.0.0.0 --port 8000

# Start Gradio UI
kataforge ui --host 0.0.0.0 --port 7860

# Show system status
kataforge status
```

## Command Reference

### Init Command

**`kataforge init`** - Set up your KataForge training space

```bash
kataforge init --name my_dojo --data-dir ~/.kataforge/data
```

**Options:**
- `--name`, `-n` - Your training space name (default: "dojo")
- `--data-dir`, `-d` - Where to store training data (default: "~/.dojo/data")

**Creates directory structure:**
```
~/.kataforge/data/
├── raw/              # Raw video footage
├── processed/        # Preprocessed videos
├── poses/            # Extracted pose data
├── models/           # Trained models
├── profiles/         # Coach profiles
├── exports/          # Exported analyses
├── checkpoints/      # Training checkpoints
└── logs/             # Training logs
```

### Extract Pose Command

**`kataforge extract-pose`** - Extract movement patterns from video for AI analysis

```bash
kataforge extract-pose technique.mp4 --output poses.json --model 2 --confidence 0.7
```

**Arguments:**
- `video` - Video file to analyze (required)

**Options:**
- `--output`, `-o` - Save pose data here (required)
- `--model`, `-m` - Detection accuracy: 0 (fast), 1 (balanced), 2 (precise) (default: 2)
- `--confidence`, `-c` - Minimum confidence for movement detection (0.0-1.0) (default: 0.7)

**Output:**
```json
{
  "total_frames": 1200,
  "fps": 30.0,
  "poses": [
    [
      [x1, y1, z1, visibility1],
      [x2, y2, z2, visibility2],
      ...
    ],
    ...
  ],
  "landmark_names": ["nose", "left_eye", "right_eye", ...]
}
```

**Example:**
```bash
# Extract poses with high accuracy
kataforge extract-pose training.mp4 -o nagato_roundhouse.json -m 2 -c 0.8

# Batch extract multiple videos
for video in *.mp4; do
  kataforge extract-pose "$video" -o "poses/${video%.mp4}.json"
done
```

### Train Command

**`kataforge train`** - Train an AI model to analyze techniques like this coach

```bash
kataforge train nagato --data-dir data/poses --epochs 100 --batch-size 16 --lr 0.001 --device auto
```

**Arguments:**
- `coach_id` - Training coach ID (required)

**Options:**
- `--data-dir`, `-d` - Training data location
- `--epochs`, `-e` - Number of training cycles (default: 100)
- `--batch-size`, `-b` - Videos processed together (default: 16)
- `--lr` - Learning speed (default: 0.001)
- `--device` - Processing power: cpu, cuda, rocm, or auto (default: auto)
- `--checkpoint-dir` - Save progress here
- `--resume`, `-r` - Continue from this checkpoint

**Training Process:**
1. Load training data from `data_dir/coach_id/`
2. Create data loaders with specified batch size
3. Initialize FormAssessor model
4. Train for specified epochs with progress updates
5. Save best model checkpoint
6. Generate training history

**Example:**
```bash
# Train with GPU acceleration
kataforge train nagato --epochs 100 --batch-size 8 --device cuda

# Resume training from checkpoint
kataforge train nagato --resume checkpoints/nagato/best_model.pt --epochs 50

# Train with ROCm (AMD GPU)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
kataforge train nagato --device rocm --batch-size 4
```

### Analyze Command

**`kataforge analyze`** - Analyze your technique and get detailed feedback

```bash
kataforge analyze technique.mp4 --coach nagato --technique roundhouse --model models/best.pt --output analysis.json --verbose
```

**Arguments:**
- `video` - Video file to analyze (required)
- `coach` - Which coach's style to match (required)
- `technique` - Name of the technique (required)

**Options:**
- `--model`, `-m` - Path to trained model
- `--output`, `-o` - Save analysis to file
- `--verbose`, `-v` - Show detailed breakdown

**Analysis Process:**
1. Extract poses from video using MediaPipe
2. Calculate biomechanics metrics
3. Load trained model for scoring
4. Generate overall and aspect scores
5. Create corrections and recommendations
6. Display results with Rich formatting
7. Save analysis if requested

**Output:**
```json
{
  "video": "technique.mp4",
  "coach": "nagato",
  "technique": "roundhouse",
  "frames": 1200,
  "fps": 30.0,
  "overall_score": 8.5,
  "aspect_scores": {
    "speed": 8.2,
    "force": 8.7,
    "timing": 7.9,
    "balance": 8.8,
    "coordination": 8.4
  },
  "biomechanics": {
    "max_speed": 4.8,
    "peak_force": 1200.5,
    "mean_power": 850.2,
    "kinetic_chain_efficiency": 88.5
  },
  "corrections": [
    "Improve hip rotation timing - initiate rotation earlier",
    "Maintain better center of gravity throughout the technique"
  ],
  "recommendations": [
    "Practice shadowboxing for 10 minutes daily",
    "Focus on chambering technique before strikes",
    "Add plyometric exercises to improve explosive speed"
  ]
}
```

**Example:**
```bash
# Basic analysis
kataforge analyze technique.mp4 --coach nagato --technique roundhouse

# Detailed analysis with model
kataforge analyze technique.mp4 -c nagato -t roundhouse -m models/nagato.pt -v

# Real-time analysis from webcam
kataforge analyze --source webcam --coach nagato --technique roundhouse
```

### Coach Commands

#### `kataforge coach add` - Add a new coach profile

```bash
kataforge coach add nagato --name "Nagato" --style "Muay Thai" --rank Champion --years 15
```

**Arguments:**
- `coach_id` - Unique coach identifier (required)

**Options:**
- `--name`, `-n` - Coach's full name (required)
- `--style`, `-s` - Martial art style (required)
- `--rank`, `-r` - Rank or belt level
- `--years`, `-y` - Years of experience (default: 0)

**Example:**
```bash
# Add multiple coaches
kataforge coach add nagato -n "Nagato" -s "Muay Thai" -r Champion -y 15
kataforge coach add sagat -n "Sagat" -s "Muay Thai" -r Master -y 25
```

#### `kataforge coach list` - List all registered coach profiles

```bash
kataforge coach list --style "Muay Thai"
```

**Options:**
- `--style`, `-s` - Filter by martial art style

**Example:**
```bash
# List all coaches
kataforge coach list

# Filter by style
kataforge coach list --style "Muay Thai"
```

#### `kataforge coach show` - Show detailed information about a coach

```bash
kataforge coach show nagato
```

**Arguments:**
- `coach_id` - Coach identifier (required)

**Example:**
```bash
# Show coach details
kataforge coach show nagato

# Get coach information for analysis
coach_info=$(kataforge coach show nagato)
```

#### `kataforge coach delete` - Delete a coach profile

```bash
kataforge coach delete nagato --force
```

**Arguments:**
- `coach_id` - Coach identifier (required)

**Options:**
- `--force`, `-f` - Skip confirmation

**Example:**
```bash
# Delete coach with confirmation
kataforge coach delete old_coach

# Force delete without confirmation
kataforge coach delete temp_coach --force
```

### Serve Command

**`kataforge serve`** - Start the API server

```bash
kataforge serve --host 0.0.0.0 --port 8000 --workers 4 --reload
```

**Options:**
- `--host`, `-h` - Host to bind to (default: 0.0.0.0)
- `--port`, `-p` - Port to bind to (default: 8000)
- `--workers`, `-w` - Number of worker processes (default: 1)
- `--reload`, `-r` - Enable auto-reload (development only)

**Environment Variables:**
```bash
# Configure via environment
export DOJO_API_HOST=0.0.0.0
export DOJO_API_PORT=8000
export DOJO_API_WORKERS=4
export DOJO_API_RELOAD=true
kataforge serve
```

**Example:**
```bash
# Development server with auto-reload
kataforge serve --reload

# Production server with multiple workers
kataforge serve --workers 4

# Custom host and port
kataforge serve --host 192.168.1.100 --port 8080
```

### UI Command

**`kataforge ui`** - Start the Gradio web interface

```bash
kataforge ui --host 0.0.0.0 --port 7860 --api-url http://localhost:8000 --ollama-url http://localhost:11434 --llm-backend ollama --share
```

**Options:**
- `--host`, `-h` - Host to bind to (default: 0.0.0.0)
- `--port`, `-p` - Port to bind to (default: 7860)
- `--api-url`, `-a` - Dojo API URL (default: http://localhost:8000)
- `--ollama-url`, `-o` - Ollama/llama.cpp URL (default: http://localhost:11434)
- `--llm-backend`, `-l` - LLM backend: ollama or llamacpp (default: ollama)
- `--share`, `-s` - Create public Gradio link

**Environment Variables:**
```bash
# Configure via environment
export DOJO_GRADIO_HOST=0.0.0.0
export DOJO_GRADIO_PORT=7860
export DOJO_GRADIO_SHARE=true
kataforge ui
```

**Example:**
```bash
# Start UI with default settings
kataforge ui

# Start UI with custom API backend
kataforge ui --api-url http://api.kataforge.com --llm-backend llamacpp

# Start UI with public sharing
kataforge ui --share
```

### Status Command

**`kataforge status`** - Show system status and configuration

```bash
kataforge status
```

**Output:**
```
╔═══════════════════════════════════════════════════════════════╗
║                        KataForge Status                        ║
╠═══════════════════════════════════════════════════════════════╣
║ Component       Value                     Status               ║
╠═══════════════════════════════════════════════════════════════╣
║ KataForge      0.1.0                     ●                    ║
║ Python         3.11.6                    ●                    ║
║ PyTorch        2.1.0 (CUDA: True)        ●                    ║
║ OpenCV         4.8.0                     ●                    ║
║ MediaPipe      0.10.0                    ●                    ║
║ FastAPI        0.104.0                   ●                    ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║                     Configuration                            ║
╠═══════════════════════════════════════════════════════════════╣
║ Setting         Value                                      ║
╠═══════════════════════════════════════════════════════════════╣
║ Environment     development                                 ║
║ Debug           False                                      ║
║ API Host        0.0.0.0                                    ║
║ API Port        8000                                       ║
║ Log Level       INFO                                        ║
║ Data Dir        /home/user/.kataforge/data                 ║
╚═══════════════════════════════════════════════════════════════╝

Registered coaches: 2
```

**Example:**
```bash
# Check system status
kataforge status

# Verify configuration
kataforge status | grep -A 10 Configuration
```

## Advanced Usage

### Complete Analysis Workflow

```bash
# 1. Initialize the system
kataforge init --name my_dojo --data-dir ~/.kataforge/data

# 2. Extract pose data from video
kataforge extract-pose training.mp4 --output analysis.json

# 3. Train models with GPU acceleration
kataforge train nagato --epochs 100 --device cuda

# 4. Analyze a technique with AI feedback
kataforge analyze technique.mp4 --coach nagato --technique roundhouse --verbose

# 5. Start API server for remote access
kataforge serve --host 0.0.0.0 --port 8000

# 6. Launch Gradio UI for interactive analysis
kataforge ui --share
```

### Real-Time Analysis

```bash
# Analyze from webcam
kataforge analyze --source webcam --coach nagato --technique roundhouse

# Continuous analysis loop
while true; do
  kataforge analyze webcam.mp4 --coach nagato --technique roundhouse
  sleep 5
  mv webcam.mp4 archive/$(date +%Y%m%d_%H%M%S).mp4
done
```

### Batch Processing

```bash
# Batch extract poses
for video in videos/*.mp4; do
  kataforge extract-pose "$video" -o "poses/${video%.mp4}.json" -m 2 -c 0.8
done

# Batch analyze techniques
for pose in poses/*.json; do
  kataforge analyze "${pose%.json}.mp4" --coach nagato --technique roundhouse 
    --output "results/${pose%.json}_analysis.json"
done
```

### Multi-GPU Training

```bash
# Train with multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
kataforge train nagato --batch-size 16 --workers 2 --device cuda

# Train with ROCm (AMD)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
kataforge train nagato --device rocm --batch-size 4

# Train with Vulkan
export ENABLE_VULKAN_COMPUTE=1
kataforge train nagato --device vulkan
```

## Configuration

### Environment Variables

```bash
# Core settings
export DOJO_ENVIRONMENT=development
export DOJO_DEBUG=true
export DOJO_DATA_DIR=~/.kataforge/data

# API Server
export DOJO_API_HOST=0.0.0.0
export DOJO_API_PORT=8000
export DOJO_API_WORKERS=4
export DOJO_API_RELOAD=true

# GPU Configuration
export DOJO_MODEL_DEVICE=auto
export DOJO_GPU_MEMORY_FRACTION=0.9

# LLM Configuration
export DOJO_LLM_BACKEND=ollama
export DOJO_OLLAMA_HOST=http://localhost:11434

# Voice System
export DOJO_TTS_ENABLED=true
export DOJO_STT_ENABLED=true
```

### Configuration File

```yaml
# ~/.config/kataforge/config.yaml
data_dir: /home/user/kataforge_data
log_level: INFO
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
llm:
  backend: ollama
  vision_model: llava:7b
  text_model: mistral:7b
gpu:
  device: auto
  memory_fraction: 0.8
```

## Error Handling

### Common Errors

**Video not found:**
```
Error: Video file not found: technique.mp4
```

**Solution:** Check file path and permissions

**GPU not available:**
```
Error: PyTorch not available or no GPU detected
```

**Solution:** Install PyTorch with GPU support or use CPU

**Invalid configuration:**
```
Error: Invalid configuration - check DOJO_* environment variables
```

**Solution:** Validate configuration with `kataforge status`

### Debug Mode

```bash
# Enable debug mode
export DOJO_DEBUG=true
kataforge analyze technique.mp4 --coach nagato --verbose

# Save debug logs
kataforge analyze technique.mp4 --coach nagato 2> debug.log

# Check detailed error information
kataforge --debug analyze technique.mp4 --coach nagato
```

## Tips and Tricks

### Performance Optimization

```bash
# Use GPU acceleration
kataforge train nagato --device cuda --batch-size 16

# Optimize for Framework 16 (AMD)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
kataforge train nagato --device rocm --batch-size 4

# Use mixed precision training
kataforge train nagato --mixed-precision true
```

### Memory Management

```bash
# Limit GPU memory usage
export DOJO_GPU_MEMORY_FRACTION=0.7
kataforge train nagato --batch-size 8

# Use smaller batch size
kataforge train nagato --batch-size 4

# Enable gradient checkpointing
kataforge train nagato --gradient-checkpointing true
```

### Advanced Analysis

```bash
# Use custom model
kataforge analyze technique.mp4 --coach nagato --model custom_model.pt

# Get detailed biomechanics
kataforge analyze technique.mp4 --coach nagato --verbose

# Compare multiple techniques
kataforge analyze technique1.mp4 --coach nagato --output analysis1.json
kataforge analyze technique2.mp4 --coach nagato --output analysis2.json
diff analysis1.json analysis2.json
```

## CLI Architecture

### Command Structure

```
kataforge [global_options] <command> [command_options] [arguments]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version`, `-V` | Show version and exit |
| `--debug` | Enable debug mode |

### Command Groups

- **Video Processing**: `init`, `extract-pose`
- **Training**: `train`
- **Analysis**: `analyze`
- **Coach Management**: `coach`
- **Server**: `serve`
- **UI**: `ui`
- **System**: `status`

## Development

### Adding New Commands

```python
# Add new command to kataforge/cli/main.py
@app.command()
def my_command(
    arg1: str = Argument(..., help="Argument help"),
    opt1: int = Option(42, "--option", help="Option help")
):
    """Command description"""
    print(f"Running with {arg1} and {opt1}")
```

### Testing Commands

```bash
# Test command directly
python -m kataforge.cli.main my_command --help

# Test with different inputs
kataforge my_command --option 123

# Test error handling
kataforge my_command invalid_input 2> error.log
```

## Troubleshooting

### Command Not Found

```bash
# Check installation
which kataforge

# Reinstall if needed
pip install --upgrade kataforge

# Check Python path
python -c "import kataforge; print(kataforge.__file__)"
```

### Permission Issues

```bash
# Check permissions
ls -la $(which kataforge)

# Fix permissions
chmod +x $(which kataforge)

# Run with sudo if needed
sudo kataforge init
```

### GPU Detection Issues

```bash
# Check GPU detection
nvidia-smi  # NVIDIA
rocm-smi    # AMD
vulkaninfo  # Vulkan

# Set GPU manually
export DOJO_MODEL_DEVICE=cuda
kataforge train nagato
```

### Dependency Issues

```bash
# Check dependencies
pip list | grep -E "(torch|opencv|mediapipe|fastapi)"

# Install missing dependencies
pip install torch opencv-python mediapipe fastapi

# Use Nix environment
nix develop
```

## Best Practices

1. **Use Nix Environment** for reproducible builds
2. **Enable Debug Mode** for troubleshooting
3. **Check System Status** before training
4. **Use GPU Acceleration** when available
5. **Monitor Resource Usage** during training
6. **Validate Configuration** before production
7. **Backup Data** regularly
8. **Use Version Control** for training data
9. **Document Workflows** for reproducibility
10. **Monitor Performance** and optimize as needed

## Performance Benchmarks

| Command | Typical Time | GPU Acceleration | Notes |
|---------|--------------|------------------|-------|
| `init` | < 1s | No | Creates directories |
| `extract-pose` | 10-30s | Yes | Depends on video length |
| `train` | 2-4 days | Yes | 100 epochs, full dataset |
| `analyze` | 2-5s | Yes | Real-time capable |
| `serve` | - | Yes | API server |
| `ui` | - | Yes | Gradio UI |
| `status` | < 1s | No | System check |

## Command Aliases

Add aliases to your shell configuration:

```bash
# ~/.bashrc or ~/.zshrc
alias kf="kataforge"
alias kf-init="kataforge init"
alias kf-extract="kataforge extract-pose"
alias kf-train="kataforge train"
alias kf-analyze="kataforge analyze"
alias kf-serve="kataforge serve"
alias kf-ui="kataforge ui"
alias kf-status="kataforge status"
```

## Integration with Other Tools

### Git Integration

```bash
# Track training data with Git
cd ~/.kataforge/data
git init
echo "*.mp4" > .gitignore
echo "*.json" >> .gitignore
git add .gitignore
```

### Docker Integration

```bash
# Run in Docker
docker run -it --gpus all \
  -v ~/.kataforge:/root/.kataforge \
  kataforge:latest \
  kataforge analyze technique.mp4 --coach nagato
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: CLI Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install kataforge
      - name: Run CLI tests
        run: kataforge status
```

## Conclusion

The KataForge CLI provides a comprehensive interface for martial arts technique analysis with support for video processing, training, analysis, coach management, and system monitoring. The CLI is built with Typer and Rich for beautiful output and includes comprehensive error handling and help documentation.

For best results:
- Use Nix environment for reproducible builds
- Enable GPU acceleration when available
- Monitor system status regularly
- Use debug mode for troubleshooting
- Follow best practices for data management
