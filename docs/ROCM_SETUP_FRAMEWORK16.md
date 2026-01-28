# ROCm Setup Guide - Framework 16 (AMD RX 7700S)

**Copyright © 2026 DeMoD LLC. All rights reserved.**

## 🎯 Overview

This guide helps you set up ROCm (Radeon Open Compute) on your Framework 16 laptop with AMD Radeon RX 7700S for training Dojo Manager ML models.

**Your Hardware:**
- **CPU**: AMD Ryzen 9 7840HS (8 cores, 16 threads, up to 5.1 GHz)
- **GPU**: AMD Radeon RX 7700S (8GB GDDR6, RDNA3, 32 CUs)
- **Architecture**: gfx1100 (RDNA3)
- **Memory**: 32GB DDR5 (recommended)
- **ROCm Version**: 6.0+

---

## 📋 Prerequisites

### 1. Check Your System

```bash
# Check CPU
lscpu | grep "Model name"

# Check GPU
lspci | grep VGA

# Check kernel version (need 6.1+)
uname -r

# Check system RAM
free -h
```

### 2. Verify GPU is Detected

```bash
# Should show your RX 7700S
lspci -k | grep -A 3 VGA

# Check if amdgpu driver is loaded
lsmod | grep amdgpu
```

---

## 🚀 Installation

### Method 1: Using Nix Flake (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/demod-llc/dojo-manager.git
cd dojo-manager

# 2. Copy ROCm flake
cp flake-rocm.nix flake.nix

# 3. Enter ROCm development environment
nix develop

# You should see:
# ╔═══════════════════════════════════════════════════════════════╗
# ║  🥋 Dojo Manager - ROCm/AMD GPU Development Environment      ║
# ║  GPU: AMD Radeon RX 7700S (8GB VRAM)                         ║
# ╚═══════════════════════════════════════════════════════════════╝

# 4. Verify PyTorch sees your GPU
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Should output:
# ROCm available: True
# GPU: AMD Radeon RX 7700S
```

### Method 2: Manual ROCm Installation

If Nix doesn't work, install ROCm manually:

#### Ubuntu/Debian

```bash
# 1. Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

# 2. Install ROCm
sudo apt install rocm-hip-sdk rocm-libs

# 3. Add user to render and video groups
sudo usermod -a -G render,video $USER

# 4. Reboot
sudo reboot

# 5. Verify installation
rocminfo
rocm-smi
```

#### Arch Linux

```bash
# ROCm is in AUR
yay -S rocm-hip-sdk rocm-opencl-runtime

# Add user to groups
sudo usermod -a -G render,video $USER

# Reboot
sudo reboot
```

---

## ⚙️ Configuration

### 1. Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# ROCm paths
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm

# Critical for RX 7700S (RDNA3)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# Stability improvements
export HSA_ENABLE_SDMA=0

# Performance tuning
export AMD_LOG_LEVEL=3
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# Library paths
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm/bin:$PATH
```

Reload:
```bash
source ~/.bashrc
```

### 2. Framework 16 Specific Optimizations

#### Power Management

```bash
# Check current power profile
cat /sys/class/drm/card1/device/power_dpm_force_performance_level

# Set to high performance for training
echo "high" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

# Or use performance mode
echo "performance" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

#### Thermal Management

```bash
# Monitor temperatures during training
watch -n 1 rocm-smi

# The RX 7700S can safely run at up to 95°C, but aim for 80-85°C
# If temps are too high:
# 1. Elevate laptop for better airflow
# 2. Use laptop cooling pad
# 3. Reduce batch size
# 4. Enable power limit
```

#### Memory Configuration

```bash
# Check VRAM usage
rocm-smi --showmeminfo vram

# For 8GB VRAM, use conservative settings:
# - Batch size: 4-8
# - Sequence length: 32-64
# - Mixed precision: FP16
# - Gradient checkpointing: enabled
```

---

## 🧪 Testing Your Setup

### 1. Basic GPU Test

Create `test_gpu.py`:

```python
#!/usr/bin/env python3
"""Test ROCm/PyTorch GPU setup"""

import torch
import sys

def test_rocm():
    print("=" * 60)
    print("ROCm/PyTorch GPU Test")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # Check ROCm availability
    if torch.cuda.is_available():
        print(f"✓ ROCm available: True")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ ROCm not available!")
        print("\nTroubleshooting:")
        print("1. Check HSA_OVERRIDE_GFX_VERSION=11.0.0 is set")
        print("2. Run: rocminfo (should show gfx1100)")
        print("3. Reinstall PyTorch with ROCm support")
        sys.exit(1)
    
    # Test tensor operations
    print("\n" + "=" * 60)
    print("Testing tensor operations...")
    print("=" * 60)
    
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Matrix multiplication
        z = torch.mm(x, y)
        
        print("✓ Matrix multiplication: SUCCESS")
        
        # Memory test
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        
        print(f"✓ Memory allocated: {allocated:.2f} GB")
        print(f"✓ Memory cached: {cached:.2f} GB")
        
    except Exception as e:
        print(f"✗ GPU operations failed: {e}")
        sys.exit(1)
    
    # Performance test
    print("\n" + "=" * 60)
    print("Performance test (1000x1000 matmul × 100)")
    print("=" * 60)
    
    import time
    
    # Warmup
    for _ in range(10):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✓ Time: {elapsed:.3f} seconds")
    print(f"✓ GFLOPS: {(2 * 1000**3 * 100) / elapsed / 1e9:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Your GPU is ready for training.")
    print("=" * 60)

if __name__ == "__main__":
    test_rocm()
```

Run the test:

```bash
python test_gpu.py
```

Expected output:
```
============================================================
ROCm/PyTorch GPU Test
============================================================

✓ PyTorch version: 2.1.0+rocm6.0
✓ ROCm available: True
✓ GPU count: 1
✓ GPU name: AMD Radeon RX 7700S
✓ GPU memory: 8.00 GB

============================================================
Testing tensor operations...
============================================================
✓ Matrix multiplication: SUCCESS
✓ Memory allocated: 0.00 GB
✓ Memory cached: 0.01 GB

============================================================
Performance test (1000x1000 matmul × 100)
============================================================
✓ Time: 0.142 seconds
✓ GFLOPS: 1408.45

============================================================
✅ All tests passed! Your GPU is ready for training.
============================================================
```

### 2. Monitor GPU During Test

In another terminal:

```bash
# Real-time monitoring
watch -n 1 rocm-smi

# Or use nvtop (works with AMD GPUs)
nvtop
```

---

## 🎓 Training Configuration for RX 7700S

### Optimized Training Config

Create `config/framework16_rocm.yaml`:

```yaml
# Training configuration optimized for Framework 16 RX 7700S

hardware:
  device: "cuda"  # PyTorch uses "cuda" for both CUDA and ROCm
  gpu_memory_gb: 8
  architecture: "RDNA3"
  compute_units: 32

training:
  # Conservative batch size for 8GB VRAM
  batch_size: 4  # Start with 4, can increase to 8 if stable
  
  # Sequence length
  sequence_length: 32  # Shorter sequences use less memory
  
  # Mixed precision training (critical for memory efficiency)
  mixed_precision: true
  precision: "fp16"  # Use FP16, not BF16 (better AMD support)
  
  # Gradient accumulation (simulate larger batches)
  gradient_accumulation_steps: 4  # Effective batch size = 4 × 4 = 16
  
  # Memory optimization
  gradient_checkpointing: true  # Trade compute for memory
  pin_memory: true
  num_workers: 4  # 7840HS has 8 cores, use 4 for data loading
  
  # Learning rate
  learning_rate: 0.0001
  warmup_steps: 500
  
  # Epochs
  epochs: 100
  early_stopping_patience: 10
  
  # Monitoring
  log_interval: 10
  eval_interval: 100
  save_interval: 1000

model:
  graphsage:
    hidden_channels: 128
    num_layers: 3
    dropout: 0.5
  
  form_assessor:
    hidden_dim: 256
    num_lstm_layers: 2
    num_attention_heads: 8
    dropout: 0.3
  
  style_encoder:
    coach_embedding_dim: 64
    technique_embedding_dim: 32

optimizer:
  type: "adamw"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: "cosine"
  t_max: 100
  eta_min: 1e-6

# Data augmentation
augmentation:
  temporal_jitter: true
  spatial_noise: 0.01
  horizontal_flip: true
  speed_variation: [0.8, 1.2]
  rotation_range: 5  # degrees

# Framework 16 specific
framework16:
  # Power management
  power_limit: 100  # Watts
  target_temp: 85   # Celsius
  
  # Cooling
  fan_curve: "performance"
  
  # Thermal throttling detection
  check_throttling: true
  throttle_threshold: 90  # Celsius
```

### Training Script with ROCm Support

Create `train_rocm.py`:

```python
#!/usr/bin/env python3
"""
Training script with ROCm/AMD GPU support
Optimized for Framework 16 RX 7700S
"""

import torch
import yaml
from pathlib import Path
import sys

def check_gpu():
    """Check GPU availability and setup"""
    if not torch.cuda.is_available():
        print("❌ GPU not available!")
        print("Run test_gpu.py for diagnostics")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("=" * 60)
    print("GPU Configuration")
    print("=" * 60)
    print(f"Device: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 60)
    
    return gpu_name, gpu_memory

def setup_training(config_path):
    """Setup training with config"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check GPU
    gpu_name, gpu_memory = check_gpu()
    
    # Verify memory is sufficient
    required_memory = config['hardware']['gpu_memory_gb']
    if gpu_memory < required_memory:
        print(f"⚠️  Warning: GPU has {gpu_memory:.1f}GB, config expects {required_memory}GB")
        print("Reducing batch size...")
        config['training']['batch_size'] = max(2, config['training']['batch_size'] // 2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable mixed precision if configured
    if config['training']['mixed_precision']:
        print("✓ Mixed precision training enabled (FP16)")
    
    # Enable gradient checkpointing if configured
    if config['training']['gradient_checkpointing']:
        print("✓ Gradient checkpointing enabled")
    
    return config, device

def monitor_gpu():
    """Monitor GPU during training"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def main():
    print("\n🥋 Dojo Manager Training - ROCm/AMD GPU\n")
    
    # Load config
    config, device = setup_training("config/framework16_rocm.yaml")
    
    # TODO: Load data, create model, train
    print("\n✅ Configuration loaded successfully!")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Sequence length: {config['training']['sequence_length']}")
    print(f"Mixed precision: {config['training']['mixed_precision']}")
    print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    
    print("\n🚀 Ready to train! Next: Implement model training loop")

if __name__ == "__main__":
    main()
```

Run:
```bash
python train_rocm.py
```

---

## 📊 Performance Expectations

### RX 7700S Capabilities

```
Architecture:        RDNA3 (gfx1100)
Compute Units:       32
Stream Processors:   2048
VRAM:               8GB GDDR6
Memory Bandwidth:    288 GB/s
FP32 Performance:    ~11 TFLOPS
FP16 Performance:    ~22 TFLOPS (with mixed precision)
```

### Training Time Estimates

**For Dojo Manager Models:**

| Model | Batch Size | Est. Time/Epoch | Est. Total (100 epochs) |
|-------|-----------|-----------------|-------------------------|
| GraphSAGE | 4 | 15-20 min | 25-33 hours |
| Form Assessor | 4 | 20-25 min | 33-42 hours |
| Style Encoder | 4 | 10-15 min | 17-25 hours |

**Total training time (all 3 models): ~75-100 hours (3-4 days)**

### Optimization Tips

1. **Start with batch size 4**, monitor VRAM usage
2. **Use mixed precision (FP16)** - 2x faster, uses less memory
3. **Enable gradient checkpointing** - trades compute for memory
4. **Use gradient accumulation** - simulate larger batches
5. **Train overnight** - models will take 8-24 hours each
6. **Monitor temperatures** - keep under 85°C for sustained performance

---

## 🔧 Troubleshooting

### Issue 1: "ROCm not available" in PyTorch

**Solution:**
```bash
# Check HSA override is set
echo $HSA_OVERRIDE_GFX_VERSION
# Should output: 11.0.0

# If not set:
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Verify GPU is detected by ROCm
rocminfo | grep gfx1100
```

### Issue 2: Out of Memory (OOM) Errors

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 2  # Start very small

# 2. Reduce sequence length
sequence_length = 16

# 3. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Clear cache between batches
torch.cuda.empty_cache()

# 5. Use smaller model
hidden_channels = 64  # Instead of 128
```

### Issue 3: Slow Training Speed

**Solutions:**
```bash
# 1. Check GPU is in performance mode
cat /sys/class/drm/card1/device/power_dpm_force_performance_level

# 2. Set to performance
echo "performance" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

# 3. Check thermal throttling
rocm-smi

# 4. Improve cooling
# - Elevate laptop
# - Use cooling pad
# - Clean dust from vents
```

### Issue 4: Kernel Crashes / Hangs

**Solutions:**
```bash
# 1. Disable SDMA (can cause instability)
export HSA_ENABLE_SDMA=0

# 2. Reduce power limit
# In training config:
power_limit: 90  # Instead of 100

# 3. Update kernel
sudo apt update && sudo apt upgrade

# 4. Check dmesg for errors
dmesg | tail -50
```

### Issue 5: Driver Issues

**Solutions:**
```bash
# 1. Check amdgpu driver is loaded
lsmod | grep amdgpu

# 2. Check for errors
dmesg | grep -i amdgpu

# 3. Reinstall drivers
sudo apt install --reinstall linux-modules-extra-$(uname -r)

# 4. Reboot
sudo reboot
```

---

## 📈 Monitoring During Training

### Terminal 1: Training

```bash
python train_rocm.py
```

### Terminal 2: GPU Monitoring

```bash
# Option 1: rocm-smi (official AMD tool)
watch -n 1 rocm-smi

# Option 2: nvtop (more visual)
nvtop

# Option 3: htop for CPU/RAM
htop
```

### Terminal 3: Temperature Monitoring

```bash
# Watch temperature
watch -n 1 "rocm-smi | grep Temperature"

# Full system sensors
watch -n 1 sensors
```

---

## 🎯 Best Practices for Framework 16

### 1. Thermal Management

```bash
# Before training:
# 1. Close other applications
# 2. Elevate laptop (books under rear edge)
# 3. Ensure good airflow
# 4. Consider external cooling pad

# During training:
# - Monitor temps every 30 minutes
# - Target: 75-85°C
# - If >90°C: reduce batch size or take breaks
```

### 2. Power Management

```bash
# Plugged in is essential for training
# Battery will throttle performance significantly

# Check power profile
powerprofilesctl get

# Set to performance
powerprofilesctl set performance
```

### 3. Data Organization

```bash
# Use fast SSD for training data
# Framework 16 has NVMe SSD - perfect!

# Structure:
~/dojo-training/
├── data/
│   ├── raw/          # Raw videos
│   ├── processed/    # Preprocessed videos
│   ├── poses/        # Extracted poses
│   └── splits/       # Train/val/test splits
├── models/           # Saved models
├── logs/            # Training logs
└── checkpoints/     # Model checkpoints
```

### 4. Training Schedule

```bash
# Recommended schedule for your hardware:

# Day 1: Data preparation
# - Preprocess videos
# - Extract poses
# - Calculate biomechanics
# Time: 4-8 hours

# Day 2-3: Train GraphSAGE
# - Start in evening
# - Let run overnight
# Time: 25-33 hours

# Day 4-5: Train Form Assessor
# - Start in evening
# - Let run overnight
# Time: 33-42 hours

# Day 6: Train Style Encoder
# - Can run during day
# Time: 17-25 hours

# Total: 5-6 days to trained models
```

---

## ✅ Ready to Train!

Your Framework 16 with RX 7700S is now configured for training!

### Next Steps:

1. **Test your setup:**
   ```bash
   python test_gpu.py
   ```

2. **Prepare your data:**
   ```bash
   dojo-manager data validate raw_videos/
   ```

3. **Start training:**
   ```bash
   python train_rocm.py
   ```

4. **Monitor progress:**
   ```bash
   # Terminal 2
   watch -n 1 rocm-smi
   ```

---

## 📞 Support

**Issues with ROCm setup?**
- Email: ml-support@demod.llc
- Include: GPU model, ROCm version, error messages

**Framework 16 specific questions?**
- Framework Community: https://community.frame.work/
- ROCm Documentation: https://rocm.docs.amd.com/

---

**Your laptop is powerful enough to train these models!** 🚀

The RX 7700S with 8GB VRAM can absolutely train the Dojo Manager models with the optimized settings provided. Just be patient - training will take 3-5 days total for all three models.

**Copyright © 2026 DeMoD LLC. All rights reserved.**
