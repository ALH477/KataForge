"""
GPU Detection and Configuration Utility
Supports both NVIDIA CUDA and AMD ROCm GPUs

Copyright (c) 2026 DeMoD LLC. All rights reserved.
"""

import sys
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GPUInfo:
    """GPU information"""
    available: bool
    backend: str  # 'cuda', 'rocm', 'cpu'
    device_count: int
    device_name: Optional[str] = None
    compute_capability: Optional[Tuple[int, int]] = None
    total_memory_gb: Optional[float] = None
    architecture: Optional[str] = None


def detect_gpu() -> GPUInfo:
    """
    Detect available GPU and return information.
    Works with both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    """
    if not TORCH_AVAILABLE:
        return GPUInfo(
            available=False,
            backend='cpu',
            device_count=0
        )
    
    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            backend='cpu',
            device_count=0
        )
    
    # Detect backend
    backend = 'rocm' if torch.version.hip else 'cuda'
    
    # Get device info
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else None
    
    # Get memory
    total_memory = None
    if device_count > 0:
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9  # Convert to GB
    
    # Get compute capability (CUDA) or architecture (ROCm)
    compute_capability = None
    architecture = None
    
    if device_count > 0:
        if backend == 'cuda':
            compute_capability = torch.cuda.get_device_capability(0)
        else:  # ROCm
            # Try to detect AMD architecture
            device_name_lower = device_name.lower() if device_name else ''
            if '7700s' in device_name_lower or '7700' in device_name_lower:
                architecture = 'RDNA3'
            elif '6000' in device_name_lower or 'vega' in device_name_lower:
                architecture = 'Vega'
            elif 'navi' in device_name_lower:
                architecture = 'RDNA'
            else:
                architecture = 'Unknown'
    
    return GPUInfo(
        available=True,
        backend=backend,
        device_count=device_count,
        device_name=device_name,
        compute_capability=compute_capability,
        total_memory_gb=total_memory,
        architecture=architecture
    )


def get_optimal_batch_size(gpu_memory_gb: float, model_type: str = 'graphsage') -> int:
    """
    Recommend optimal batch size based on GPU memory.
    
    Args:
        gpu_memory_gb: GPU memory in GB
        model_type: Type of model ('graphsage', 'form_assessor', 'style_encoder')
    
    Returns:
        Recommended batch size
    """
    # Base memory requirements (GB per sample)
    memory_per_sample = {
        'graphsage': 0.5,      # Graph model
        'form_assessor': 1.0,  # LSTM + attention
        'style_encoder': 0.3,  # Simple encoder
    }
    
    base_memory = memory_per_sample.get(model_type, 0.5)
    
    # Account for model parameters and optimizer state (~ 2GB)
    available_memory = gpu_memory_gb - 2.0
    
    if available_memory <= 0:
        return 1
    
    # Calculate batch size
    batch_size = int(available_memory / base_memory)
    
    # Clamp to reasonable range
    batch_size = max(1, min(batch_size, 32))
    
    # Round down to power of 2 for efficiency
    import math
    power = int(math.log2(batch_size))
    batch_size = 2 ** power
    
    return batch_size


def configure_pytorch_for_gpu(gpu_info: GPUInfo) -> Dict[str, any]:
    """
    Configure PyTorch based on detected GPU.
    
    Returns:
        Configuration dictionary
    """
    config = {
        'device': 'cpu',
        'mixed_precision': False,
        'num_workers': 0,
        'pin_memory': False,
        'recommended_batch_size': 16,
    }
    
    if not gpu_info.available:
        return config
    
    # Set device
    config['device'] = 'cuda'
    
    # Enable mixed precision for compatible GPUs
    if gpu_info.backend == 'cuda':
        # CUDA: Check compute capability
        if gpu_info.compute_capability and gpu_info.compute_capability[0] >= 7:
            config['mixed_precision'] = True
    elif gpu_info.backend == 'rocm':
        # ROCm: Enable for RDNA2+ architectures
        if gpu_info.architecture in ['RDNA3', 'RDNA2', 'CDNA2']:
            config['mixed_precision'] = True
    
    # Set data loader workers
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    config['num_workers'] = min(cpu_count // 2, 8)
    
    # Enable pin_memory for faster data transfer
    config['pin_memory'] = True
    
    # Recommend batch size based on GPU memory
    if gpu_info.total_memory_gb:
        config['recommended_batch_size'] = get_optimal_batch_size(
            gpu_info.total_memory_gb
        )
    
    return config


def setup_environment_for_gpu(gpu_info: GPUInfo):
    """
    Set environment variables for optimal GPU performance.
    """
    if not gpu_info.available:
        return
    
    if gpu_info.backend == 'rocm':
        # ROCm-specific optimizations
        
        # For RDNA3 (RX 7700S, RX 7900 XT, etc.)
        if gpu_info.architecture == 'RDNA3':
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
            os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
        
        # Stability improvements
        os.environ.setdefault('HSA_ENABLE_SDMA', '0')
        
        # Performance tuning
        os.environ.setdefault('AMD_LOG_LEVEL', '3')
        os.environ.setdefault('GPU_MAX_HEAP_SIZE', '100')
        os.environ.setdefault('GPU_MAX_ALLOC_PERCENT', '100')
        
    elif gpu_info.backend == 'cuda':
        # CUDA-specific optimizations
        
        # Enable TF32 for Ampere+ GPUs
        if gpu_info.compute_capability and gpu_info.compute_capability[0] >= 8:
            if TORCH_AVAILABLE:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking
        if TORCH_AVAILABLE:
            torch.backends.cudnn.benchmark = True


def print_gpu_info(gpu_info: GPUInfo):
    """Print GPU information in a nice format."""
    print("=" * 70)
    print("GPU Configuration")
    print("=" * 70)
    
    if not gpu_info.available:
        print("❌ No GPU detected - will use CPU")
        print("\nFor training, a GPU is highly recommended.")
        print("Expected training time on CPU: 10-50x slower than GPU")
        print("=" * 70)
        return
    
    print(f"✓ GPU Available: {gpu_info.backend.upper()}")
    print(f"✓ Device Count: {gpu_info.device_count}")
    
    if gpu_info.device_name:
        print(f"✓ Device Name: {gpu_info.device_name}")
    
    if gpu_info.total_memory_gb:
        print(f"✓ GPU Memory: {gpu_info.total_memory_gb:.2f} GB")
    
    if gpu_info.backend == 'cuda' and gpu_info.compute_capability:
        print(f"✓ Compute Capability: {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}")
    
    if gpu_info.backend == 'rocm' and gpu_info.architecture:
        print(f"✓ Architecture: {gpu_info.architecture}")
    
    # Get recommended config
    config = configure_pytorch_for_gpu(gpu_info)
    
    print(f"\n📊 Recommended Training Configuration:")
    print(f"  • Batch Size: {config['recommended_batch_size']}")
    print(f"  • Mixed Precision: {config['mixed_precision']}")
    print(f"  • Data Loader Workers: {config['num_workers']}")
    print(f"  • Pin Memory: {config['pin_memory']}")
    
    print("=" * 70)


def get_device():
    """
    Get PyTorch device.
    Convenience function for training scripts.
    
    Returns:
        torch.device object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    gpu_info = detect_gpu()
    
    if gpu_info.available:
        # Setup environment
        setup_environment_for_gpu(gpu_info)
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def test_gpu_operations():
    """
    Test GPU with actual tensor operations.
    
    Returns:
        True if tests pass, False otherwise
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not installed")
        return False
    
    gpu_info = detect_gpu()
    
    if not gpu_info.available:
        print("❌ No GPU available for testing")
        return False
    
    try:
        print("\n" + "=" * 70)
        print("Testing GPU Operations")
        print("=" * 70)
        
        # Create test tensors
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Test matrix multiplication
        print("Testing matrix multiplication...")
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        
        print("✓ Matrix multiplication: PASSED")
        
        # Test memory allocation
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"✓ Memory allocated: {allocated:.3f} GB")
        print(f"✓ Memory reserved: {reserved:.3f} GB")
        
        # Performance test
        print("\nPerformance test (1000x1000 matmul × 100)...")
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
        
        gflops = (2 * 1000**3 * 100) / elapsed / 1e9
        
        print(f"✓ Time: {elapsed:.3f} seconds")
        print(f"✓ Performance: {gflops:.2f} GFLOPS")
        
        print("=" * 70)
        print("✅ All GPU tests passed!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        print("\nTroubleshooting:")
        if gpu_info.backend == 'rocm':
            print("1. Check HSA_OVERRIDE_GFX_VERSION is set correctly")
            print("2. Run: rocminfo")
            print("3. Verify ROCm version matches PyTorch")
        else:
            print("1. Check CUDA drivers are installed")
            print("2. Run: nvidia-smi")
            print("3. Verify CUDA version matches PyTorch")
        print("=" * 70)
        return False


if __name__ == "__main__":
    """Run as standalone script to test GPU"""
    
    # Detect GPU
    gpu_info = detect_gpu()
    
    # Print info
    print_gpu_info(gpu_info)
    
    # Test operations
    if gpu_info.available:
        test_gpu_operations()
    
    # Print PyTorch info
    if TORCH_AVAILABLE:
        print(f"\n📦 PyTorch version: {torch.__version__}")
        if torch.version.hip:
            print(f"📦 ROCm version: {torch.version.hip}")
        elif torch.version.cuda:
            print(f"📦 CUDA version: {torch.version.cuda}")
