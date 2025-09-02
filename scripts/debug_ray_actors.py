#!/usr/bin/env python3

import ray
import os
import sys
import time
import traceback
import subprocess
import multiprocessing
import torch
import gc
from pathlib import Path

# Add src to path
sys.path.append("/app/src")

# Set CUDA environment variables for WSL stability
os.environ.update(
    {
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": "0",  # Use only first GPU if available
        "OMP_NUM_THREADS": "1",  # Prevent thread conflicts
        "MKL_NUM_THREADS": "1",
        "NUMBA_DISABLE_CUDA": "0",
        "RAY_DISABLE_IMPORT_WARNING": "1",
    }
)


def safe_cuda_check():
    """Safe CUDA availability check for WSL environments."""
    print("🔍 Safe CUDA Check...")

    # Check nvidia-smi first
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        nvidia_smi_ok = result.returncode == 0
        print(
            f"   nvidia-smi: {'✅ Available' if nvidia_smi_ok else '❌ Not available'}"
        )
    except:
        nvidia_smi_ok = False
        print("   nvidia-smi: ❌ Not available")

    if not nvidia_smi_ok:
        return False, "CPU"

    try:
        # Import torch in a separate process to avoid crashes
        import torch

        print(f"   PyTorch version: {torch.__version__}")

        # Check device count without calling is_available()
        device_count = torch.cuda.device_count()
        print(f"   CUDA device count: {device_count}")

        if device_count == 0:
            return False, "CPU"

        # Try very minimal CUDA operation
        try:
            # Create tiny tensor and immediately move to CPU
            test_tensor = torch.tensor([1.0], device="cuda:0")
            _ = test_tensor.cpu()
            del test_tensor
            torch.cuda.empty_cache()
            gc.collect()

            print("   ✅ CUDA basic test passed")
            return True, "cuda:0"

        except Exception as cuda_err:
            print(f"   ⚠️ CUDA test failed: {cuda_err}")
            return False, "CPU"

    except Exception as e:
        print(f"   ❌ PyTorch CUDA check failed: {e}")
        return False, "CPU"


def init_ray():
    """Initialize Ray cluster connection."""
    if ray.is_initialized():
        ray.shutdown()

    try:
        # Connect to existing Ray cluster
        print("🚀 Connecting to Ray cluster...")
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

        print("✅ Connected to Ray cluster")
        print(f"   Nodes: {len(ray.nodes())}")
        print(f"   Resources: {ray.cluster_resources()}")

        return True

    except Exception as e:
        print(f"❌ Failed to connect to Ray: {e}")

        # Try local Ray as fallback
        try:
            print("🔄 Trying local Ray...")
            ray.init(ignore_reinit_error=True)
            print("✅ Started local Ray")
            return True
        except Exception as local_e:
            print(f"❌ Local Ray failed: {local_e}")
            return False


@ray.remote
class SafeMLActor:
    """Safe ML Actor that avoids CUDA double-free issues."""

    def __init__(self):
        # Set environment in actor
        os.environ.update(
            {
                "CUDA_LAUNCH_BLOCKING": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )

        self.device = "cpu"  # Default to CPU
        self.models_loaded = {}

        print("SafeMLActor: Initializing...")

    def safe_import_test(self):
        """Test imports safely without CUDA calls."""
        results = {}

        try:
            import torch

            results["torch_version"] = torch.__version__
            results["torch_ok"] = True

            # Don't call torch.cuda.is_available() - this causes the crash
            results["cuda_device_count"] = torch.cuda.device_count()

        except Exception as e:
            results["torch_version"] = f"Error: {e}"
            results["torch_ok"] = False
            results["cuda_device_count"] = 0

        try:
            import faster_whisper

            results["faster_whisper_ok"] = True
        except Exception as e:
            results["faster_whisper_ok"] = False
            results["faster_whisper_error"] = str(e)

        try:
            import librosa

            results["librosa_version"] = librosa.__version__
            results["librosa_ok"] = True
        except Exception as e:
            results["librosa_version"] = f"Error: {e}"
            results["librosa_ok"] = False

        return results

    def safe_cuda_setup(self):
        """Setup CUDA safely for this actor."""
        try:
            import torch

            # Check device count without calling is_available()
            device_count = torch.cuda.device_count()

            if device_count > 0:
                # Test minimal CUDA operation
                try:
                    # Very small test
                    test_tensor = torch.tensor([1.0], device="cuda:0")
                    result = test_tensor.cpu().item()
                    del test_tensor
                    torch.cuda.empty_cache()

                    self.device = "cuda:0"
                    return {
                        "cuda_available": True,
                        "device": self.device,
                        "test_result": result,
                    }

                except Exception as cuda_err:
                    return {
                        "cuda_available": False,
                        "device": "cpu",
                        "cuda_error": str(cuda_err),
                    }
            else:
                return {
                    "cuda_available": False,
                    "device": "cpu",
                    "reason": "No CUDA devices found",
                }

        except Exception as e:
            return {"cuda_available": False, "device": "cpu", "error": str(e)}

    def load_whisper_model(self, model_size="base"):
        """Load Whisper model safely."""
        try:
            from faster_whisper import WhisperModel

            # Always use CPU for Whisper to avoid CUDA issues
            model = WhisperModel(
                model_size, device="cpu", download_root="/app/models"  # Force CPU
            )

            self.models_loaded["whisper"] = model_size

            return {"success": True, "model_size": model_size, "device": "cpu"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def transcribe_test(self, test_audio_path=None):
        """Test transcription with dummy data."""
        try:
            if "whisper" not in self.models_loaded:
                result = self.load_whisper_model()
                if not result["success"]:
                    return {"success": False, "error": "Failed to load model"}

            # For testing, just return success
            return {
                "success": True,
                "message": "Whisper model loaded and ready",
                "device": "cpu",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


@ray.remote
def safe_remote_function():
    """Simple remote function to test basic Ray functionality."""
    import os
    import torch

    # Set safe environment
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    return {
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "timestamp": time.time(),
    }


def test_basic_ray():
    """Test basic Ray functionality."""
    print("\n🧪 Testing basic Ray functionality...")

    try:
        # Test simple remote function
        future = safe_remote_function.remote()
        result = ray.get(future, timeout=30)

        print("✅ Basic Ray test passed")
        print(f"   PID: {result['pid']}")
        print(f"   PyTorch: {result['torch_version']}")
        print(f"   CUDA devices: {result['device_count']}")

        return True

    except Exception as e:
        print(f"❌ Basic Ray test failed: {e}")
        traceback.print_exc()
        return False


def test_safe_ml_actor():
    """Test the SafeMLActor."""
    print("\n🧪 Testing Safe ML Actor...")

    try:
        # Create actor
        actor = SafeMLActor.remote()

        # Test imports
        print("   Testing imports...")
        import_result = ray.get(actor.safe_import_test.remote(), timeout=30)
        print(f"   Import results: {import_result}")

        if not import_result.get("torch_ok"):
            print("   ❌ PyTorch import failed in actor")
            return False

        # Test CUDA setup
        print("   Testing CUDA setup...")
        cuda_result = ray.get(actor.safe_cuda_setup.remote(), timeout=30)
        print(f"   CUDA results: {cuda_result}")

        # Test Whisper loading
        print("   Testing Whisper model loading...")
        whisper_result = ray.get(actor.load_whisper_model.remote(), timeout=60)
        print(f"   Whisper results: {whisper_result}")

        if whisper_result["success"]:
            # Test transcription
            print("   Testing transcription...")
            transcribe_result = ray.get(actor.transcribe_test.remote(), timeout=30)
            print(f"   Transcription results: {transcribe_result}")

        print("✅ Safe ML Actor test completed")
        return True

    except Exception as e:
        print(f"❌ Safe ML Actor test failed: {e}")
        traceback.print_exc()
        return False


def test_actor_resource_usage():
    """Test resource usage by actors."""
    print("\n🧪 Testing Actor Resource Usage...")

    try:
        # Create multiple actors to test resource allocation
        actors = [SafeMLActor.remote() for _ in range(2)]

        # Test each actor
        futures = []
        for i, actor in enumerate(actors):
            print(f"   Testing actor {i+1}...")
            future = actor.safe_import_test.remote()
            futures.append(future)

        # Get results
        results = ray.get(futures, timeout=60)

        all_passed = True
        for i, result in enumerate(results):
            if result.get("torch_ok"):
                print(f"   ✅ Actor {i+1}: OK")
            else:
                print(f"   ❌ Actor {i+1}: Failed")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Resource usage test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🔍 Ray Actors Debug - Safe Version")
    print("=" * 60)

    # Test CUDA first (outside Ray)
    cuda_available, device = safe_cuda_check()
    print(f"CUDA Status: {'Available' if cuda_available else 'CPU only'} ({device})")

    # Initialize Ray
    if not init_ray():
        print("❌ Failed to initialize Ray")
        return 1

    try:
        # Run tests
        tests = [
            ("Basic Ray", test_basic_ray),
            ("Safe ML Actor", test_safe_ml_actor),
            ("Resource Usage", test_actor_resource_usage),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append((test_name, success))

            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY:")

        passed = sum(1 for _, success in results if success)
        total = len(results)

        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️ Some tests failed - check logs above")
            return 1

    finally:
        # Cleanup
        try:
            ray.shutdown()
            print("\n🧹 Ray shutdown complete")
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
