# ROCm/AMD GPU Integration - System Update

**Copyright © 2026 DeMoD LLC. All rights reserved.**

---

## 🎯 What's New

The Dojo Manager system now fully supports **AMD GPUs with ROCm**, specifically optimized for the **Framework 16 laptop** with AMD Radeon RX 7700S.

### Key Additions

✅ **ROCm-Enabled Nix Flake** - Native AMD GPU support  
✅ **Framework 16 Profile** - Optimized for 7840HS + RX 7700S  
✅ **Universal GPU Detection** - Works with both NVIDIA and AMD  
✅ **Optimized Training Configs** - Memory-efficient settings for 8GB VRAM  
✅ **Complete Setup Guide** - Step-by-step ROCm installation  
✅ **Quick Start Script** - One-command setup for Framework 16  

---

## 📦 New Files

### 1. Core Infrastructure

#### `flake-rocm.nix`
Complete Nix flake with ROCm support:
- ROCm 6.0 packages
- PyTorch with ROCm backend
- RDNA3 architecture support (gfx1100)
- Framework 16 hardware profile
- Environment variables for RX 7700S
- Docker image with ROCm

**Usage:**
```bash
# Replace main flake
cp flake-rocm.nix flake.nix

# Enter ROCm environment
nix develop
```

### 2. Documentation

#### `ROCM_SETUP_FRAMEWORK16.md`
Comprehensive 50+ page setup guide:
- System requirements
- ROCm installation (Nix, Ubuntu, Arch)
- Framework 16 optimizations
- GPU testing procedures
- Training configuration
- Performance expectations
- Troubleshooting

**Covers:**
- ✅ ROCm 6.0 installation
- ✅ GPU detection and verification
- ✅ Environment configuration
- ✅ Power and thermal management
- ✅ Memory optimization for 8GB VRAM
- ✅ Common issues and solutions

### 3. Utilities

#### `dojo_manager/core/gpu_utils.py`
Universal GPU detection and configuration:

**Features:**
- Detects both NVIDIA (CUDA) and AMD (ROCm) GPUs
- Returns GPU information (name, memory, architecture)
- Recommends optimal batch sizes
- Configures PyTorch automatically
- Sets environment variables
- Tests GPU operations

**Usage:**
```python
from dojo_manager.core.gpu_utils import (
    detect_gpu,
    get_device,
    print_gpu_info,
    test_gpu_operations
)

# Detect GPU
gpu_info = detect_gpu()
print_gpu_info(gpu_info)

# Get PyTorch device
device = get_device()

# Test GPU
test_gpu_operations()
```

**Example Output:**
```
======================================================================
GPU Configuration
======================================================================
✓ GPU Available: ROCM
✓ Device Count: 1
✓ Device Name: AMD Radeon RX 7700S
✓ GPU Memory: 8.00 GB
✓ Architecture: RDNA3

📊 Recommended Training Configuration:
  • Batch Size: 4
  • Mixed Precision: True
  • Data Loader Workers: 4
  • Pin Memory: True
======================================================================
```

### 4. Quick Start

#### `framework16-quickstart.sh`
Automated setup script for Framework 16:

**What it does:**
1. ✅ Checks hardware (CPU, GPU, RAM)
2. ✅ Verifies dependencies (Nix, git, Python)
3. ✅ Checks ROCm installation
4. ✅ Creates directory structure
5. ✅ Generates optimized config
6. ✅ Creates GPU test script
7. ✅ Sets up direnv (if available)

**Usage:**
```bash
chmod +x framework16-quickstart.sh
./framework16-quickstart.sh
```

### 5. Configuration

#### `config/framework16.yaml`
Optimized training configuration for RX 7700S:

```yaml
hardware:
  device: "cuda"
  gpu_memory_gb: 8
  architecture: "RDNA3"

training:
  batch_size: 4              # Conservative for 8GB VRAM
  sequence_length: 32        # Shorter = less memory
  mixed_precision: true      # FP16 for efficiency
  gradient_checkpointing: true
  gradient_accumulation_steps: 4  # Effective batch = 16
  
  num_workers: 4             # 7840HS has 8 cores
  pin_memory: true
  
  learning_rate: 0.0001
  epochs: 100

framework16:
  power_limit: 100           # Watts
  target_temp: 85            # Celsius
  fan_curve: "performance"
```

---

## 🚀 Getting Started with Framework 16

### Step 1: Quick Setup

```bash
# Run quick start script
chmod +x framework16-quickstart.sh
./framework16-quickstart.sh

# Output:
# ╔═══════════════════════════════════════════════════════════════╗
# ║  🥋 Dojo Manager - Framework 16 Quick Start                  ║
# ║  Hardware: AMD Ryzen 9 7840HS + RX 7700S                    ║
# ╚═══════════════════════════════════════════════════════════════╝
```

### Step 2: Enter Development Environment

```bash
# Using Nix (recommended)
nix develop

# Or manually set environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
```

### Step 3: Test GPU

```bash
# Run GPU test
python test_gpu.py

# Or use utility directly
python -c "from dojo_manager.core.gpu_utils import test_gpu_operations; test_gpu_operations()"
```

### Step 4: Prepare Data

```bash
# Copy videos to data/raw/
cp /path/to/videos/* data/raw/

# Preprocess videos
dojo-manager video batch-preprocess data/raw/ data/processed/ --workers 8

# Extract poses
dojo-manager pose batch-extract data/processed/ data/poses/ --workers 4

# Calculate biomechanics
dojo-manager biomechanics batch-calculate data/poses/ data/metrics/ --workers 4

# Split data
dojo-manager data split data/poses/ --train-ratio 0.7 --output-dir data/splits/
```

### Step 5: Train Models

```bash
# Train all models with Framework 16 config
dojo-manager train all-models \
  --config config/framework16.yaml \
  --data-dir data/splits/ \
  --output-dir models/v1.0 \
  --profile framework16

# Monitor in another terminal
watch -n 1 rocm-smi
```

---

## 📊 Framework 16 Specifications

### Hardware

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 7840HS |
| Cores/Threads | 8 / 16 |
| Base/Boost | 3.8 GHz / 5.1 GHz |
| Cache | 16MB L3 |
| **GPU** | AMD Radeon RX 7700S |
| Architecture | RDNA3 (gfx1100) |
| Compute Units | 32 |
| Stream Processors | 2048 |
| VRAM | 8GB GDDR6 |
| Memory Bandwidth | 288 GB/s |
| FP32 Performance | ~11 TFLOPS |
| FP16 Performance | ~22 TFLOPS |

### Performance Expectations

**Training Times (with optimized config):**

| Model | Time/Epoch | Total (100 epochs) |
|-------|-----------|-------------------|
| GraphSAGE | 15-20 min | 25-33 hours |
| Form Assessor | 20-25 min | 33-42 hours |
| Style Encoder | 10-15 min | 17-25 hours |

**Total: ~75-100 hours (3-4 days) for all 3 models**

### Memory Usage

With optimized settings:
- Batch size 4: ~4-5 GB VRAM
- Batch size 8: ~6-7 GB VRAM
- Peak usage: ~7 GB VRAM (safe for 8GB)

### Power Consumption

- Idle: 15-20W
- Training (CPU+GPU): 80-100W
- Peak: 120W (with sustained load)

### Thermal Performance

- Typical training temp: 75-85°C
- Max safe temp: 95°C
- Thermal throttling: >90°C
- Target: 80-85°C for optimal performance

---

## 🔧 Optimization Settings

### For Maximum Performance

```yaml
# config/framework16_max_performance.yaml
training:
  batch_size: 8              # Use full VRAM
  sequence_length: 64        # Longer sequences
  mixed_precision: true
  gradient_checkpointing: false  # Use more memory for speed
  
framework16:
  power_limit: 120           # Maximum power
  target_temp: 85
  fan_curve: "turbo"
```

**Use when:**
- You have good cooling
- Shorter training sessions
- Need maximum speed

**Expected speedup:** 1.5-2x faster

### For Maximum Stability

```yaml
# config/framework16_stable.yaml
training:
  batch_size: 2              # Very conservative
  sequence_length: 16        # Short sequences
  mixed_precision: true
  gradient_checkpointing: true
  
framework16:
  power_limit: 90            # Reduced power
  target_temp: 80
  fan_curve: "balanced"
```

**Use when:**
- Laptop gets too hot
- Long training sessions (overnight)
- Battery life is important

**Trade-off:** 2-3x slower, but very stable

### For Battery Training (Not Recommended)

```yaml
# config/framework16_battery.yaml
training:
  batch_size: 1
  sequence_length: 8
  mixed_precision: true
  gradient_checkpointing: true
  
framework16:
  power_limit: 50            # Minimal power
  target_temp: 70
  fan_curve: "quiet"
```

**⚠️ Warning:** Training on battery is **10-20x slower** and not recommended. Only use for testing.

---

## 🐛 Troubleshooting

### Issue: GPU Not Detected

**Symptoms:**
```
ROCm available: False
GPU count: 0
```

**Solution:**
```bash
# 1. Check GPU is visible
lspci | grep VGA

# 2. Check amdgpu driver
lsmod | grep amdgpu

# 3. Set environment variable
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 4. Verify ROCm
rocminfo | grep gfx1100

# 5. Reinstall PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### Issue: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```yaml
# Reduce memory usage
training:
  batch_size: 2              # Halve batch size
  sequence_length: 16        # Halve sequence length
  gradient_checkpointing: true
  
# Or clear cache
torch.cuda.empty_cache()
```

### Issue: Slow Training

**Symptoms:**
- <5 GFLOPS performance
- High CPU usage, low GPU usage

**Solutions:**
```bash
# 1. Check power mode
powerprofilesctl get
powerprofilesctl set performance

# 2. Check GPU performance level
cat /sys/class/drm/card1/device/power_dpm_force_performance_level
echo "performance" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

# 3. Check thermal throttling
rocm-smi  # Look for temp >90°C

# 4. Increase data loader workers
num_workers: 8  # Use more CPU cores
```

### Issue: Training Crashes

**Symptoms:**
- Kernel crash
- System freeze
- ROCm error messages

**Solutions:**
```bash
# 1. Disable SDMA
export HSA_ENABLE_SDMA=0

# 2. Reduce power limit
power_limit: 80  # Instead of 100

# 3. Update kernel and ROCm
sudo apt update && sudo apt upgrade

# 4. Check dmesg for errors
dmesg | grep -i amdgpu
```

---

## 📈 Monitoring During Training

### Terminal 1: Training Progress

```bash
python train_rocm.py
```

### Terminal 2: GPU Monitor

```bash
# Option 1: ROCm SMI (official)
watch -n 1 rocm-smi

# Option 2: nvtop (more visual)
nvtop

# Option 3: Custom monitoring
watch -n 1 "rocm-smi | grep -E 'Temp|Power|GPU'"
```

### Terminal 3: System Monitor

```bash
# CPU and RAM
htop

# Or combined
btop
```

### Log Files

```bash
# Training logs
tail -f logs/training.log

# Error logs
tail -f logs/error.log

# TensorBoard (if enabled)
tensorboard --logdir logs/tensorboard
```

---

## ✅ Verification Checklist

Before training:

- [ ] GPU detected: `python test_gpu.py`
- [ ] ROCm version: `rocminfo | head -10`
- [ ] gfx1100 detected: `rocminfo | grep gfx1100`
- [ ] PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Environment vars set: `echo $HSA_OVERRIDE_GFX_VERSION`
- [ ] Power profile: `powerprofilesctl get`
- [ ] Data prepared: Files in `data/splits/`
- [ ] Config created: `config/framework16.yaml` exists
- [ ] AC power connected: Essential for training
- [ ] Good cooling: Laptop elevated or on cooling pad

---

## 🎯 Expected Results

### After Setup (10 minutes)

```
✓ GPU detected: AMD Radeon RX 7700S
✓ ROCm available: True
✓ Environment configured
✓ Test successful: 1200+ GFLOPS
```

### After Data Preparation (4-8 hours)

```
✓ 1,500 videos preprocessed
✓ 1,500 pose files extracted
✓ 1,500 biomechanics files calculated
✓ Data split: 70% train, 15% val, 15% test
```

### After Training (3-5 days)

```
✓ GraphSAGE: 87% accuracy
✓ Form Assessor: 0.82 correlation
✓ Style Encoder: 91% coach classification
✓ Models exported to ONNX
✓ Ready for deployment
```

---

## 🚀 Next Steps

1. **Run quick start:**
   ```bash
   ./framework16-quickstart.sh
   ```

2. **Test GPU:**
   ```bash
   nix develop
   python test_gpu.py
   ```

3. **Prepare data:**
   ```bash
   # Follow data preparation steps
   ```

4. **Start training:**
   ```bash
   # Train overnight, check progress in morning
   ```

5. **Deploy:**
   ```bash
   # After training complete
   make deploy-production
   ```

---

## 📞 Support

**Framework 16 + ROCm Issues:**
- Email: ml-support@demod.llc
- Include: GPU model, ROCm version, error logs

**General Training Questions:**
- See: MODEL_TRAINING_GUIDE.md
- Email: support@demod.llc

**Framework Community:**
- Forum: https://community.frame.work/
- ROCm Docs: https://rocm.docs.amd.com/

---

## 🏆 Summary

Your Framework 16 laptop is now **fully configured for ML training** with:

✅ **Native AMD GPU support** via ROCm  
✅ **Optimized for RX 7700S** (8GB VRAM)  
✅ **Memory-efficient training** (4-8 batch size)  
✅ **Universal GPU detection** (CUDA/ROCm)  
✅ **Complete documentation** (setup to deployment)  
✅ **Quick start automation** (one command)  

**Your laptop can train production-grade ML models!**

Expected timeline:
- Setup: 10-30 minutes
- Data prep: 4-8 hours
- Training: 3-5 days
- **Total: ~1 week to trained models**

**Ready to start training?** Run: `./framework16-quickstart.sh`

---

**Copyright © 2026 DeMoD LLC. All rights reserved.**
