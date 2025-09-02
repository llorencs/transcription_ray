#!/usr/bin/env python3
"""
Debug Ray actors specifically - simulate the exact same flow as transcription service.
"""

import ray
import sys
import os
from pathlib import Path

# Add src to path (same as the transcription service does)
sys.path.append("/app/src")


def test_exact_ray_flow():
    """Test the exact same Ray flow as the transcription service."""

    print("🔍 Ray Actors Debug - Simulating Transcription Flow")
    print("=" * 60)

    try:
        # Step 1: Initialize Ray exactly like the service does
        print("1. Initializing Ray connection...")
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            print("✅ Ray initialized")
        else:
            print("✅ Ray already initialized")

        print(f"   Cluster resources: {ray.cluster_resources()}")

        # Step 2: Import Ray actors (this is where it might fail)
        print("\n2. Testing Ray actor imports...")
        try:
            from src.services.ray_actors import (
                TranscriptionCoordinator,
                WhisperTranscriptionActor,
            )

            print("✅ Ray actor imports successful")
        except ImportError as e:
            print(f"❌ Ray actor import failed: {e}")
            return False

        # Step 3: Create a simple actor (like WhisperTranscriptionActor does)
        print("\n3. Testing simple Ray actor creation...")

        @ray.remote
        def test_simple_imports():
            """Simple test that mimics what WhisperTranscriptionActor does."""
            import sys

            sys.path.append("/app/src")

            results = {}
            try:
                import torch

                results["torch"] = f"SUCCESS - {torch.__version__}"
            except ImportError as e:
                results["torch"] = f"FAILED - {str(e)}"

            try:
                from faster_whisper import WhisperModel

                results["faster_whisper"] = "SUCCESS"
            except ImportError as e:
                results["faster_whisper"] = f"FAILED - {str(e)}"

            return results

        future = test_simple_imports.remote()
        result = ray.get(future)

        print("   Results from simple actor:")
        for lib, status in result.items():
            print(f"     {lib}: {status}")

        if "FAILED" in str(result):
            print("❌ Simple actor test failed")
            return False

        # Step 4: Test WhisperTranscriptionActor specifically
        print("\n4. Testing WhisperTranscriptionActor creation...")

        try:
            whisper_actor = WhisperTranscriptionActor.remote(
                model_size="base", device="cpu"
            )
            print("✅ WhisperTranscriptionActor created successfully")

            # Test language detection (simpler than full transcription)
            print("5. Testing language detection method...")

            # We need a test audio file - create a dummy one if needed
            test_audio = "/app/temp/test_dummy.wav"

            # Create a very short dummy audio file for testing
            try:
                import numpy as np
                import soundfile as sf

                # Generate 1 second of silence
                sample_rate = 16000
                duration = 1.0
                samples = np.zeros(int(sample_rate * duration))

                os.makedirs("/app/temp", exist_ok=True)
                sf.write(test_audio, samples, sample_rate)
                print(f"✅ Created test audio file: {test_audio}")

            except Exception as e:
                print(f"⚠️ Could not create test audio: {e}")
                print("   Skipping audio test")
                return True  # Dependencies work, just no test file

            # Test the actor method
            try:
                lang_future = whisper_actor.detect_language.remote(test_audio)
                lang_result = ray.get(lang_future, timeout=30)
                print(f"✅ Language detection worked: {lang_result}")

            except Exception as e:
                print(f"❌ Language detection failed: {e}")
                import traceback

                traceback.print_exc()
                return False

        except Exception as e:
            print(f"❌ WhisperTranscriptionActor creation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\n✅ All Ray actor tests passed!")
        return True

    except Exception as e:
        print(f"❌ Ray flow test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_exact_ray_flow()
    sys.exit(0 if success else 1)
