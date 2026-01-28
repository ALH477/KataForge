# GPU Setup Guide

## Supported Backends
- ROCm (AMD)
- CUDA (NVIDIA)
- Vulkan (Intel/AMD)

## Installation
Use Nix flakes for reproducible environments:

```bash
# Enter environment
nix develop .#rocm

# Verify GPU detection
dojo-manager status
```

## ROCm Configuration
Required environment variables:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_PATH=/opt/rocm
```

## CUDA Configuration
Requires:
- CUDA 12.x installed
- `libcuda.so` in LD_LIBRARY_PATH

## Vulkan Configuration
For Intel GPUs:
```bash
export ENABLE_VULKAN_COMPUTE=1
export VULKAN_SDK=/usr/local/vulkan
```

See flake.nix for exact dependencies.