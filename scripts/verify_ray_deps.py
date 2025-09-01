#!/usr/bin/env python3
"""
Verify that ML dependencies are available in the Ray container.
"""


def verify_ray_imports():
    """Verify that all required ML libraries can be imported in Ray."""

    print("🔍 Verifying ML dependencies in Ray container...")
    print("=" * 60)

    success = True

    # Test basic imports
    imports_to_test = [
        ("torch", "PyTorch"),
        ("faster_whisper", "Faster Whisper"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("ray", "Ray"),
        ("pyannote.audio", "Pyannote Audio"),
    ]

    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}: Available")
        except ImportError as e:
            print(f"❌ {display_name}: NOT AVAILABLE - {e}")
            success = False

    print("=" * 60)

    if success:
        print("🎉 All ML dependencies are available in Ray!")

        # Test more specific functionality
        print("\n🧪 Testing specific Ray functionality...")

        try:
            import torch

            print(f"✅ PyTorch version: {torch.__version__}")
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU count: {torch.cuda.device_count()}")
        except Exception as e:
            print(f"⚠️ PyTorch test failed: {e}")

        try:
            from faster_whisper import WhisperModel

            print("✅ Faster Whisper: Can import WhisperModel")

            # Try to create a model on CPU (safer for testing)
            print("🤖 Testing Whisper model loading...")
            import os

            # Create models directory
            os.makedirs("/app/models/whisper", exist_ok=True)

            # Try to load the base model on CPU
            model = WhisperModel("base", device="cpu", download_root="/app/models")
            print("✅ Faster Whisper: Successfully loaded base model on CPU")

        except Exception as e:
            print(f"⚠️ Faster Whisper test failed: {e}")

        try:
            import librosa

            print(f"✅ Librosa version: {librosa.__version__}")
        except Exception as e:
            print(f"⚠️ Librosa test failed: {e}")

        try:
            import ray

            print(f"✅ Ray version: {ray.__version__}")

            # Check if Ray is initialized
            if ray.is_initialized():
                print("✅ Ray is initialized")
                print(f"   Cluster resources: {ray.cluster_resources()}")
            else:
                print("⚠️ Ray is not initialized (this is normal for verification)")

        except Exception as e:
            print(f"⚠️ Ray test failed: {e}")

        print("\n🎯 Ray environment is ready for ML processing!")

    else:
        print("❌ ML dependencies are missing in Ray container!")
        print("\nTo fix this:")
        print("1. Make sure you built the Ray container with ML dependencies:")
        print("   docker compose build ray-head --no-cache")
        print("2. Check that requirements.ray.txt includes all ML libraries")
        print("3. Verify the Dockerfile.ray includes the ML requirements")

    return success


if __name__ == "__main__":
    import sys

    success = verify_ray_imports()
    sys.exit(0 if success else 1)
