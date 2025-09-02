#!/usr/bin/env python3
"""
Debug Ray actors specifically - simulate the exact same flow as transcription service.
This version executes inside the Ray container where all dependencies should be available.
"""

import os
import sys
from pathlib import Path


def test_exact_ray_flow():
    """Test the exact same Ray flow as the transcription service."""

    print("🔍 Ray Actors Debug - Simulating Transcription Flow")
    print("=" * 60)

    try:
        # Add paths like the service does
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")

        # Step 1: Test imports
        print("1. Testing basic imports...")
        try:
            import torch

            print(f"   ✅ torch: {torch.__version__}")
        except ImportError as e:
            print(f"   ❌ torch: {e}")
            return False

        try:
            import ray

            print(f"   ✅ ray: {ray.__version__}")
        except ImportError as e:
            print(f"   ❌ ray: {e}")
            return False

        try:
            from faster_whisper import WhisperModel

            print(f"   ✅ faster_whisper: OK")
        except ImportError as e:
            print(f"   ❌ faster_whisper: {e}")
            return False

        try:
            import librosa

            print(f"   ✅ librosa: {librosa.__version__}")
        except ImportError as e:
            print(f"   ❌ librosa: {e}")
            return False

        # Step 2: Initialize Ray exactly like the service does
        print("\n2. Initializing Ray connection...")
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            print("✅ Ray initialized")
        else:
            print("✅ Ray already initialized")

        print(f"   Cluster resources: {ray.cluster_resources()}")

        # Step 3: Import Ray actors
        print("\n3. Testing Ray actor imports...")
        try:
            from src.services.ray_actors import (
                TranscriptionCoordinator,
                WhisperTranscriptionActor,
            )

            print("✅ Ray actor imports successful")
        except ImportError as e:
            print(f"❌ Ray actor import failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 4: Create a simple test function
        print("\n4. Testing simple Ray remote function...")

        @ray.remote
        def test_ml_imports_remote():
            """Test ML imports inside Ray worker."""
            import sys

            sys.path.insert(0, "/app")
            sys.path.insert(0, "/app/src")

            results = {}
            try:
                import torch

                results["torch"] = f"SUCCESS - {torch.__version__}"
                results["torch_cuda"] = torch.cuda.is_available()
            except ImportError as e:
                results["torch"] = f"FAILED - {str(e)}"
                results["torch_cuda"] = False

            try:
                from faster_whisper import WhisperModel

                results["faster_whisper"] = "SUCCESS"
            except ImportError as e:
                results["faster_whisper"] = f"FAILED - {str(e)}"

            try:
                import librosa

                results["librosa"] = f"SUCCESS - {librosa.__version__}"
            except ImportError as e:
                results["librosa"] = f"FAILED - {str(e)}"

            try:
                import soundfile

                results["soundfile"] = "SUCCESS"
            except ImportError as e:
                results["soundfile"] = f"FAILED - {str(e)}"

            return results

        future = test_ml_imports_remote.remote()
        result = ray.get(future, timeout=30)

        print("   Results from Ray worker:")
        for lib, status in result.items():
            print(f"     {lib}: {status}")

        if any("FAILED" in str(status) for status in result.values()):
            print("❌ Some imports failed in Ray worker")
            return False

        # Step 5: Test WhisperTranscriptionActor specifically
        print("\n5. Testing WhisperTranscriptionActor creation...")

        try:
            whisper_actor = WhisperTranscriptionActor.remote(
                model_size="base", device="cpu"
            )
            print("✅ WhisperTranscriptionActor created successfully")

            # Test getting status
            status_future = whisper_actor.get_status.remote()
            status_result = ray.get(status_future, timeout=60)
            print(f"   Actor status: {status_result}")

            if not status_result.get("initialized"):
                print("❌ WhisperTranscriptionActor failed to initialize properly")
                return False

        except Exception as e:
            print(f"❌ WhisperTranscriptionActor creation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 6: Create test audio file
        print("\n6. Creating test audio file...")
        try:
            import numpy as np
            import soundfile as sf

            # Generate 2 seconds of simple sine wave
            sample_rate = 16000
            duration = 2.0
            frequency = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)

            test_audio_path = "/app/temp/test_debug.wav"
            os.makedirs("/app/temp", exist_ok=True)
            sf.write(test_audio_path, audio_data, sample_rate)
            print(f"✅ Created test audio: {test_audio_path}")

        except Exception as e:
            print(f"⚠️ Could not create test audio: {e}")
            print("   Skipping audio test")
            return True  # Dependencies work, just no test file

        # Step 7: Test language detection
        print("\n7. Testing language detection...")
        try:
            lang_future = whisper_actor.detect_language.remote(test_audio_path)
            lang_result = ray.get(lang_future, timeout=120)  # 2 minutes timeout
            print(f"✅ Language detection worked: {lang_result}")

        except Exception as e:
            print(f"❌ Language detection failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 8: Test TranscriptionCoordinator
        print("\n8. Testing TranscriptionCoordinator...")
        try:
            coordinator = TranscriptionCoordinator.remote()

            # Get coordinator status
            coord_status_future = coordinator.get_status.remote()
            coord_status = ray.get(coord_status_future, timeout=30)
            print(f"   Coordinator status: {coord_status}")

            # Test transcription pipeline
            print("   Testing transcription pipeline...")
            transcription_future = coordinator.process_transcription.remote(
                audio_path=test_audio_path,
                model_size="base",
                language=None,
                initial_prompt=None,
                diarize=False,
                use_gpu=False,  # Use CPU for testing
                max_segment_duration=600.0,
            )

            transcription_result = ray.get(
                transcription_future, timeout=300
            )  # 5 minutes
            print(f"✅ Full transcription pipeline worked!")
            print(
                f"   Language: {transcription_result.get('transcription', {}).get('language', 'N/A')}"
            )
            print(
                f"   Segments: {len(transcription_result.get('transcription', {}).get('segments', []))}"
            )
            print(
                f"   Words: {len(transcription_result.get('transcription', {}).get('words', []))}"
            )

        except Exception as e:
            print(f"❌ TranscriptionCoordinator test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Cleanup
        try:
            if os.path.exists(test_audio_path):
                os.unlink(test_audio_path)
        except:
            pass

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
