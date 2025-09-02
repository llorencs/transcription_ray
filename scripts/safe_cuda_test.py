#!/usr/bin/env python3
"""
Safe CUDA test that won't crash in WSL.
"""

import os
import sys
import subprocess


def safe_cuda_test():
    """Test CUDA availability without crashing the process."""

    print("🔍 Safe CUDA Test for WSL")
    print("=" * 40)

    # Test 1: Check if nvidia-smi works
    print("1. Testing nvidia-smi...")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("   ✅ nvidia-smi working")
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    print(f"   GPU: {line.strip()}")
            nvidia_available = True
        else:
            print("   ❌ nvidia-smi failed")
            nvidia_available = False
    except Exception as e:
        print(f"   ❌ nvidia-smi error: {e}")
        nvidia_available = False

    # Test 2: Import PyTorch safely
    print("\n2. Testing PyTorch import...")
    try:
        import torch

        pytorch_version = torch.__version__
        print(f"   ✅ PyTorch {pytorch_version} imported")
    except Exception as e:
        print(f"   ❌ PyTorch import failed: {e}")
        return False

    # Test 3: Safe CUDA check using environment variables
    print("\n3. Safe CUDA availability check...")

    # Set conservative environment for WSL
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    cuda_available = False
    try:
        # Try a very minimal CUDA operation
        if nvidia_available:
            # Check if CUDA runtime is available without calling torch.cuda.is_available()
            try:
                # This is safer than torch.cuda.is_available() in WSL
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    print(f"   ✅ CUDA runtime detected: {device_count} device(s)")
                    cuda_available = True
                else:
                    print("   ⚠️ CUDA runtime available but no devices")
            except RuntimeError as cuda_error:
                print(f"   ⚠️ CUDA runtime error: {cuda_error}")
                print("   Will fall back to CPU mode")
            except Exception as e:
                print(f"   ⚠️ CUDA check failed: {e}")
                print("   Will fall back to CPU mode")
        else:
            print("   ⚠️ nvidia-smi not available, assuming no CUDA")

    except Exception as e:
        print(f"   ❌ CUDA test failed: {e}")

    # Test 4: Safe tensor creation
    print("\n4. Testing tensor operations...")
    try:
        # CPU tensor (always works)
        cpu_tensor = torch.randn(10, 10)
        print(f"   ✅ CPU tensor: {cpu_tensor.shape}")

        if cuda_available:
            try:
                # Very small CUDA tensor test
                cuda_tensor = torch.randn(5, 5, device="cuda:0")
                result = cuda_tensor.cpu()
                print(f"   ✅ CUDA tensor test passed: {result.shape}")
            except Exception as cuda_e:
                print(f"   ⚠️ CUDA tensor test failed: {cuda_e}")
                cuda_available = False

    except Exception as e:
        print(f"   ❌ Tensor operations failed: {e}")
        return False

    # Summary
    print("\n📊 Summary:")
    print(f"   PyTorch version: {pytorch_version}")
    print(
        f"   NVIDIA drivers: {'✅ Available' if nvidia_available else '❌ Not available'}"
    )
    print(f"   CUDA support: {'✅ Working' if cuda_available else '⚠️ CPU only'}")

    if cuda_available:
        print("   🎯 Ready for GPU acceleration")
    else:
        print("   💻 CPU-only mode (will be slower but functional)")

    return True


def get_safe_device():
    """Get a safe device string for PyTorch operations."""
    try:
        import torch

        # Check nvidia-smi first
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            nvidia_ok = result.returncode == 0
        except:
            nvidia_ok = False

        if nvidia_ok:
            try:
                # Safe CUDA check
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Test with a tiny operation
                    test_tensor = torch.randn(2, 2, device="cuda:0")
                    _ = test_tensor.cpu()
                    return "cuda"
            except:
                pass

        return "cpu"

    except:
        return "cpu"


if __name__ == "__main__":
    safe_cuda_test()
