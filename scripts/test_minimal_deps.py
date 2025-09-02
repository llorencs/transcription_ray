#!/usr/bin/env python3
"""
Test minimal dependencies needed for transcription.
This should run inside the Ray container.
"""

import sys


def test_minimal_deps():
    """Test only the essential dependencies."""

    print("🧪 Testing Minimal Dependencies")
    print("=" * 40)

    success_count = 0
    total_tests = 0

    # Test 1: PyTorch
    total_tests += 1
    try:
        import torch

        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        success_count += 1
    except ImportError as e:
        print(f"❌ torch: {e}")

    # Test 2: Ray
    total_tests += 1
    try:
        import ray

        print(f"✅ ray: {ray.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ ray: {e}")

    # Test 3: Faster Whisper
    total_tests += 1
    try:
        from faster_whisper import WhisperModel

        print(f"✅ faster_whisper: OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ faster_whisper: {e}")

    # Test 4: Librosa
    total_tests += 1
    try:
        import librosa

        print(f"✅ librosa: {librosa.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ librosa: {e}")

    # Test 5: SoundFile
    total_tests += 1
    try:
        import soundfile

        print(f"✅ soundfile: OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ soundfile: {e}")

    # Test 6: NumPy
    total_tests += 1
    try:
        import numpy

        print(f"✅ numpy: {numpy.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ numpy: {e}")

    print("=" * 40)
    print(f"📊 Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All essential dependencies available!")
        return True
    else:
        print("❌ Some essential dependencies missing!")
        return False


if __name__ == "__main__":
    success = test_minimal_deps()
    sys.exit(0 if success else 1)
