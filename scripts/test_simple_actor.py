#!/usr/bin/env python3
"""
Test very simple Ray actor to diagnose the actor death issue.
"""

import ray
import sys
import os
import traceback

# Add paths
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")


def test_simple_actor():
    """Test progressively more complex actors."""

    print("🧪 Testing Simple Ray Actors")
    print("=" * 50)

    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            print("✅ Ray initialized")

        # Test 1: Ultra-simple actor
        print("\n1. Testing ultra-simple actor...")

        @ray.remote
        class SimpleActor:
            def __init__(self):
                self.value = 42
                print("SimpleActor initialized")

            def get_value(self):
                return self.value

        try:
            simple_actor = SimpleActor.remote()
            result = ray.get(simple_actor.get_value.remote(), timeout=30)
            print(f"✅ Ultra-simple actor works: {result}")
        except Exception as e:
            print(f"❌ Ultra-simple actor failed: {e}")
            traceback.print_exc()
            return False

        # Test 2: Actor with imports
        print("\n2. Testing actor with basic imports...")

        @ray.remote
        class ImportActor:
            def __init__(self):
                try:
                    import torch

                    self.torch_version = torch.__version__
                    print(f"ImportActor: torch {self.torch_version}")

                    # Safe CUDA detection for WSL
                    import os
                    import subprocess

                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

                    # Check nvidia-smi first
                    try:
                        subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
                        nvidia_ok = True
                        print("ImportActor: nvidia-smi available")
                    except:
                        nvidia_ok = False
                        print("ImportActor: nvidia-smi not available")

                    # Safe CUDA check
                    self.cuda_available = False
                    if nvidia_ok:
                        try:
                            device_count = torch.cuda.device_count()
                            if device_count > 0:
                                # Very small test
                                test_tensor = torch.randn(2, 2, device="cuda:0")
                                _ = test_tensor.cpu()
                                self.cuda_available = True
                                print("ImportActor: CUDA test passed")
                        except Exception as e:
                            print(f"ImportActor: CUDA test failed: {e}")

                except Exception as e:
                    self.torch_version = f"Error: {e}"
                    self.cuda_available = False
                    print(f"ImportActor: torch import failed: {e}")

            def get_imports(self):
                results = {}

                try:
                    import torch

                    results["torch"] = self.torch_version
                    results["torch_cuda"] = self.cuda_available
                except Exception as e:
                    results["torch"] = f"Error: {e}"
                    results["torch_cuda"] = False

                try:
                    import numpy

                    results["numpy"] = numpy.__version__
                except Exception as e:
                    results["numpy"] = f"Error: {e}"

                return results

        try:
            import_actor = ImportActor.remote()
            result = ray.get(import_actor.get_imports.remote(), timeout=30)
            print(f"✅ Import actor works: {result}")
        except Exception as e:
            print(f"❌ Import actor failed: {e}")
            traceback.print_exc()
            return False

        # Test 3: Actor that tries to load Whisper (the problematic one)
        print("\n3. Testing Whisper-like actor...")

        @ray.remote
        class WhisperTestActor:
            def __init__(self):
                print("WhisperTestActor: Starting initialization...")
                self.model = None
                self.initialized = False

                try:
                    print("WhisperTestActor: Testing torch import...")
                    import torch

                    print(f"WhisperTestActor: torch OK - {torch.__version__}")

                    print("WhisperTestActor: Testing faster_whisper import...")
                    from faster_whisper import WhisperModel

                    print("WhisperTestActor: faster_whisper import OK")

                    print("WhisperTestActor: Creating model (CPU only for safety)...")
                    # Create models directory
                    import os

                    os.makedirs("/app/models/whisper", exist_ok=True)

                    self.model = WhisperModel(
                        "base",
                        device="cpu",  # Force CPU for testing
                        compute_type="int8",
                        download_root="/app/models/whisper",
                    )
                    print("WhisperTestActor: Model created successfully!")
                    self.initialized = True

                except Exception as e:
                    print(f"WhisperTestActor: Initialization failed: {e}")
                    import traceback

                    traceback.print_exc()
                    self.initialized = False
                    raise

            def get_status(self):
                return {
                    "initialized": self.initialized,
                    "model_loaded": self.model is not None,
                }

        try:
            print("   Creating WhisperTestActor...")
            whisper_actor = WhisperTestActor.remote()
            print("   Getting status...")
            result = ray.get(
                whisper_actor.get_status.remote(), timeout=120
            )  # 2 minutes timeout
            print(f"✅ Whisper test actor works: {result}")
        except Exception as e:
            print(f"❌ Whisper test actor failed: {e}")
            traceback.print_exc()

            # Try to get more info about the actor
            try:
                print("   Trying to get actor state...")
                actor_state = ray.util.state.get_actor("WhisperTestActor")
                print(f"   Actor state: {actor_state}")
            except Exception as state_e:
                print(f"   Could not get actor state: {state_e}")

            return False

        print("\n✅ All actor tests passed!")
        return True

    except Exception as e:
        print(f"❌ Actor test failed with exception: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_actor()
    sys.exit(0 if success else 1)
