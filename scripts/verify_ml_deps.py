#!/usr/bin/env python3
"""
Verify that ML dependencies are available in the API container.
"""


def verify_imports():
    """Verify that all required ML libraries can be imported."""

    print("🔍 Verifying ML dependencies in API container...")
    print("=" * 60)

    success = True

    # Test basic imports
    imports_to_test = [
        ("torch", "PyTorch"),
        ("faster_whisper", "Faster Whisper"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
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
        print("🎉 All ML dependencies are available!")

        # Test more specific functionality
        print("\n🧪 Testing specific functionality...")

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

            # Try to create a model (this will test if it can actually load)
            print("🤖 Testing model loading...")
            import tempfile
            import os

            # Create a temp directory for models
            temp_models_dir = "/app/models/whisper"
            os.makedirs(temp_models_dir, exist_ok=True)

            # Try to load the base model on CPU (safer test)
            model = WhisperModel("base", device="cpu", download_root="/app/models")
            print("✅ Faster Whisper: Successfully loaded base model on CPU")

        except Exception as e:
            print(f"⚠️ Faster Whisper test failed: {e}")

        try:
            import librosa

            print(f"✅ Librosa version: {librosa.__version__}")
        except Exception as e:
            print(f"⚠️ Librosa test failed: {e}")

        print("\n🎯 ML environment is ready for direct transcription!")

    else:
        print("❌ ML dependencies are missing!")
        print("\nTo fix this:")
        print("1. Make sure you built the API container with ML dependencies:")
        print("   docker compose build api --no-cache")
        print("2. Check that requirements.api.txt includes ML libraries")
        print("3. Verify the Dockerfile.api includes the ML requirements")

    return success


if __name__ == "__main__":
    import sys

    success = verify_imports()
    sys.exit(0 if success else 1)
