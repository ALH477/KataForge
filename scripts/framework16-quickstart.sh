#!/bin/bash
# Quick Start Script for Framework 16 with AMD RX 7700S
# Copyright (c) 2026 DeMoD LLC. All rights reserved.

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🥋 Dojo Manager - Framework 16 Quick Start                  ║"
echo "║                                                               ║"
echo "║  Hardware: AMD Ryzen 9 7840HS + RX 7700S                    ║"
echo "║  ROCm Support: AMD GPU Training                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

# Check system
print_step "Checking system..."

# Check if Framework 16
if lspci | grep -i "AMD.*7700S" &> /dev/null; then
    echo -e "${GREEN}✓${NC} AMD RX 7700S detected"
else
    print_warning "RX 7700S not detected - this script is optimized for Framework 16"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CPU
if lscpu | grep "7840HS" &> /dev/null; then
    echo -e "${GREEN}✓${NC} AMD Ryzen 9 7840HS detected"
else
    print_warning "Different CPU detected"
fi

echo ""

# Check dependencies
print_step "Checking dependencies..."

ALL_DEPS_OK=true

check_command "nix" || ALL_DEPS_OK=false
check_command "git" || ALL_DEPS_OK=false
check_command "python3" || ALL_DEPS_OK=false

if [ "$ALL_DEPS_OK" = false ]; then
    print_error "Missing required dependencies"
    echo ""
    echo "Install Nix: sh <(curl -L https://nixos.org/nix/install) --daemon"
    echo "Enable flakes: mkdir -p ~/.config/nix && echo 'experimental-features = nix-command flakes' >> ~/.config/nix/nix.conf"
    exit 1
fi

echo ""

# Check ROCm
print_step "Checking ROCm..."

if command -v rocminfo &> /dev/null; then
    echo -e "${GREEN}✓${NC} ROCm installed"
    
    # Check for gfx1100 (RDNA3)
    if rocminfo | grep "gfx1100" &> /dev/null; then
        echo -e "${GREEN}✓${NC} RDNA3 (gfx1100) detected"
    else
        print_warning "gfx1100 not detected - HSA_OVERRIDE_GFX_VERSION may be needed"
    fi
else
    print_warning "ROCm not detected"
    echo "The Nix environment will provide ROCm support"
fi

echo ""

# Setup directory
print_step "Setting up Dojo Manager..."

if [ ! -d "dojo-manager" ]; then
    echo "Directory structure:"
    mkdir -p dojo-manager
    cd dojo-manager
    
    # Create basic structure
    mkdir -p data/{raw,processed,poses,metrics,splits}
    mkdir -p models
    mkdir -p logs
    mkdir -p config
    
    echo -e "${GREEN}✓${NC} Created directory structure"
else
    cd dojo-manager
    echo -e "${GREEN}✓${NC} Using existing directory"
fi

echo ""

# Copy configuration
print_step "Creating Framework 16 configuration..."

cat > config/framework16.yaml << 'EOF'
# Training configuration for Framework 16 RX 7700S

hardware:
  device: "cuda"  # PyTorch uses "cuda" for both CUDA and ROCm
  gpu_memory_gb: 8
  architecture: "RDNA3"
  compute_units: 32

training:
  # Conservative settings for 8GB VRAM
  batch_size: 4
  sequence_length: 32
  
  # Essential for memory efficiency
  mixed_precision: true
  precision: "fp16"
  gradient_checkpointing: true
  
  # Simulate larger batches
  gradient_accumulation_steps: 4
  
  # Data loading
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  
  # Training parameters
  learning_rate: 0.0001
  epochs: 100
  early_stopping_patience: 10
  
  # Checkpointing
  save_interval: 1000
  eval_interval: 100

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

framework16:
  power_limit: 100
  target_temp: 85
  fan_curve: "performance"
EOF

echo -e "${GREEN}✓${NC} Created config/framework16.yaml"

echo ""

# Create test script
print_step "Creating GPU test script..."

cat > test_gpu.py << 'EOF'
#!/usr/bin/env python3
"""Test GPU setup for Framework 16"""

import sys

try:
    from dojo_manager.core.gpu_utils import (
        detect_gpu,
        print_gpu_info,
        test_gpu_operations
    )
    
    # Detect GPU
    gpu_info = detect_gpu()
    
    # Print info
    print_gpu_info(gpu_info)
    
    # Test operations
    if gpu_info.available:
        success = test_gpu_operations()
        sys.exit(0 if success else 1)
    else:
        print("\n⚠️  No GPU detected")
        print("Training will be very slow on CPU")
        sys.exit(1)
        
except ImportError:
    print("Setting up environment...")
    print("Run: nix develop")
    sys.exit(1)
EOF

chmod +x test_gpu.py
echo -e "${GREEN}✓${NC} Created test_gpu.py"

echo ""

# Create .envrc for direnv
if command -v direnv &> /dev/null; then
    print_step "Setting up direnv..."
    
    cat > .envrc << 'EOF'
use flake

# ROCm environment for Framework 16
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export AMD_LOG_LEVEL=3
EOF
    
    echo -e "${GREEN}✓${NC} Created .envrc"
    echo "Run 'direnv allow' to enable automatic environment loading"
fi

echo ""

# Summary
print_step "Setup complete!"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  📋 Next Steps                                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Enter the development environment:"
echo "   ${GREEN}nix develop${NC}"
echo ""
echo "2. Test your GPU:"
echo "   ${GREEN}python test_gpu.py${NC}"
echo ""
echo "3. Prepare your training data:"
echo "   ${GREEN}# Copy videos to data/raw/${NC}"
echo "   ${GREEN}dojo-manager video batch-preprocess data/raw/ data/processed/${NC}"
echo "   ${GREEN}dojo-manager pose batch-extract data/processed/ data/poses/${NC}"
echo ""
echo "4. Train your first model:"
echo "   ${GREEN}dojo-manager train all-models --config config/framework16.yaml${NC}"
echo ""
echo "5. Monitor GPU during training:"
echo "   ${GREEN}watch -n 1 rocm-smi${NC}"
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  📚 Documentation                                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "• Complete setup guide: ROCM_SETUP_FRAMEWORK16.md"
echo "• Training guide: MODEL_TRAINING_GUIDE.md"
echo "• Architecture docs: ARCHITECTURE.md"
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ⚡ Performance Tips                                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "• Plug in your laptop (AC power required for full performance)"
echo "• Set power profile: powerprofilesctl set performance"
echo "• Elevate laptop rear for better cooling"
echo "• Close other applications during training"
echo "• Training takes ~3-5 days for all 3 models"
echo ""
echo "🎯 Your Framework 16 is ready to train ML models!"
echo ""
