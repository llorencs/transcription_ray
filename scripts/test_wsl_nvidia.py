#!/usr/bin/env python3
"""
Test NVIDIA GPU functionality in WSL environment.
This should run inside the Ray container.
"""


def test_wsl_nvidia():
    """Test NVIDIA GPU setup in WSL."""

    print("🔍 Testing WSL + NVIDIA Setup")
    print("=" * 40)

    # Test 1: Basic CUDA availability
    try:
        import torch

        print(f"✅ PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")

        if cuda_available:
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"     Memory: {props.total_memory / (1024**3):.1f} GB")
                print(f"     Compute capability: {props.major}.{props.minor}")
        else:
            print("   ⚠️ CUDA not available")

    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

    # Test 2: CUDA computation if available
    if cuda_available:
        try:
            print("\n🧮 Testing CUDA computation...")

            # Small computation test
            a = torch.randn(1000, 1000, device="cuda")
            b = torch.randn(1000, 1000, device="cuda")
            c = torch.matmul(a, b)
            result = c.cpu().numpy()

            print(f"   ✅ CUDA computation successful: {result.shape}")

            # Memory test
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            memory_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            print(
                f"   GPU memory: {memory_allocated:.1f} MB allocated, {memory_reserved:.1f} MB reserved"
            )

        except Exception as e:
            print(f"   ❌ CUDA computation failed: {e}")
            print("   This might indicate a WSL/NVIDIA driver issue")
            return False

    # Test 3: Check for common WSL issues
    print("\n🔍 Checking WSL-specific issues...")

    try:
        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ nvidia-smi available")
            # Parse GPU info from nvidia-smi
            lines = result.stdout.split("\n")
            for line in lines:
                if "RTX" in line or "GTX" in line or "ADA" in line:
                    print(f"   GPU info: {line.strip()}")
        else:
            print("   ⚠️ nvidia-smi not available or failed")
    except Exception as e:
        print(f"   ⚠️ Could not run nvidia-smi: {e}")

    # Test 4: Environment variables
    print("\n🌍 Checking environment...")
    import os

    cuda_home = os.environ.get("CUDA_HOME", "Not set")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "Not set")

    print(f"   CUDA_HOME: {cuda_home}")
    print(f"   LD_LIBRARY_PATH: {ld_library_path}")

    if "/usr/local/cuda" not in ld_library_path:
        print("   ⚠️ CUDA library path might not be set correctly")

    print("\n📊 Summary:")
    if cuda_available:
        print("   ✅ GPU setup appears to be working")
        print("   🎯 Ready for GPU-accelerated transcription")
        return True
    else:
        print("   ⚠️ GPU not available - will fall back to CPU")
        print("   💡 CPU transcription will work but be slower")
        return True  # Still return True as CPU mode is acceptable


if __name__ == "__main__":
    test_wsl_nvidia()
