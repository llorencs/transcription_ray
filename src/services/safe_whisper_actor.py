#!/usr/bin/env python3
"""
Safe Whisper Actor that avoids CUDA double free issues.
"""

import ray
import os
import sys
import torch
import gc
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.append("/app/src")

# CRITICAL: Set environment before any CUDA operations
os.environ.update(
    {
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_DISABLE_CUDA": "0",
    }
)


@ray.remote(num_cpus=1, memory=2048 * 1024 * 1024)  # 2GB memory limit
class SafeWhisperActor:
    """
    Safe Whisper actor that avoids CUDA double free errors.

    Key safety features:
    - Always uses CPU for Whisper (most stable)
    - Safe CUDA detection without torch.cuda.is_available()
    - Proper memory management and cleanup
    - Conservative resource allocation
    """

    def __init__(self):
        """Initialize the actor with safe defaults."""

        # Set environment in actor process
        os.environ.update(
            {
                "CUDA_LAUNCH_BLOCKING": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )

        self.whisper_model = None
        self.device = "cpu"  # Always use CPU for Whisper
        self.model_size = None
        self.models_dir = Path("/app/models")
        self.models_dir.mkdir(exist_ok=True)

        print(f"SafeWhisperActor initialized (PID: {os.getpid()})")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information safely."""

        info = {
            "pid": os.getpid(),
            "device": self.device,
            "model_loaded": self.whisper_model is not None,
            "model_size": self.model_size,
        }

        try:
            import torch

            info["torch_version"] = torch.__version__

            # Safe device count check (don't call is_available())
            info["cuda_device_count"] = torch.cuda.device_count()

            # Memory info
            import psutil

            process = psutil.Process()
            info["memory_mb"] = process.memory_info().rss / 1024 / 1024

        except Exception as e:
            info["error"] = str(e)

        return info

    def load_model(
        self, model_size: str = "base", force_cpu: bool = True
    ) -> Dict[str, Any]:
        """
        Load Whisper model safely.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
            force_cpu: Always use CPU (recommended for stability)
        """

        try:
            print(f"Loading Whisper model: {model_size}")

            from faster_whisper import WhisperModel

            # Always use CPU to avoid CUDA issues
            device = "cpu" if force_cpu else self.device

            # Load model with conservative settings
            model = WhisperModel(
                model_size,
                device=device,
                download_root=str(self.models_dir),
                compute_type="int8" if device == "cpu" else "float16",
            )

            # Clean up any existing model
            if self.whisper_model is not None:
                del self.whisper_model
                gc.collect()

            self.whisper_model = model
            self.model_size = model_size
            self.device = device

            print(f"✅ Model loaded: {model_size} on {device}")

            return {
                "success": True,
                "model_size": model_size,
                "device": device,
                "compute_type": "int8" if device == "cpu" else "float16",
            }

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Model loading failed: {error_msg}")
            traceback.print_exc()

            return {"success": False, "error": error_msg, "model_size": model_size}

    def transcribe_file(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file safely.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional transcription parameters
        """

        if self.whisper_model is None:
            return {
                "success": False,
                "error": "Model not loaded. Call load_model() first.",
            }

        try:
            audio_path = Path(audio_path)

            if not audio_path.exists():
                return {
                    "success": False,
                    "error": f"Audio file not found: {audio_path}",
                }

            print(f"Transcribing: {audio_path}")

            # Transcription parameters with safe defaults
            transcribe_params = {
                "beam_size": 5,
                "language": None,
                "task": "transcribe",
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500, threshold=0.5),
            }

            # Update with provided kwargs
            transcribe_params.update(kwargs)

            # Perform transcription
            segments, info = self.whisper_model.transcribe(
                str(audio_path), **transcribe_params
            )

            # Collect results
            transcript_segments = []
            full_text = []

            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }
                transcript_segments.append(segment_data)
                full_text.append(segment.text.strip())

            result = {
                "success": True,
                "file": str(audio_path),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "full_text": " ".join(full_text),
                "segments": transcript_segments,
                "model_size": self.model_size,
                "device": self.device,
            }

            print(f"✅ Transcription completed: {len(transcript_segments)} segments")
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Transcription failed: {error_msg}")
            traceback.print_exc()

            return {
                "success": False,
                "error": error_msg,
                "file": str(audio_path) if "audio_path" in locals() else "unknown",
            }

    def transcribe_with_diarization(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe with speaker diarization (CPU only for safety).

        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
        """

        try:
            # First, do regular transcription
            transcription_result = self.transcribe_file(audio_path, **kwargs)

            if not transcription_result["success"]:
                return transcription_result

            # For now, return transcription without diarization
            # Diarization can be added later when stable
            transcription_result["diarization"] = "Not implemented (for stability)"
            transcription_result["speakers"] = []

            return transcription_result

        except Exception as e:
            return {
                "success": False,
                "error": f"Diarization failed: {str(e)}",
                "file": audio_path,
            }

    def cleanup(self) -> Dict[str, Any]:
        """Clean up resources."""

        try:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.device_count() > 0:
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore CUDA cleanup errors

            print("✅ Cleanup completed")

            return {"success": True, "message": "Resources cleaned up"}

        except Exception as e:
            return {"success": False, "error": str(e)}


def test_safe_whisper_actor():
    """Test the SafeWhisperActor."""

    print("🧪 Testing SafeWhisperActor...")

    try:
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

        # Create actor
        actor = SafeWhisperActor.remote()

        # Test system info
        print("1. Getting system info...")
        info = ray.get(actor.get_system_info.remote())
        print(f"   System info: {info}")

        # Test model loading
        print("2. Loading Whisper model...")
        load_result = ray.get(actor.load_model.remote("base", force_cpu=True))
        print(f"   Load result: {load_result}")

        if load_result["success"]:
            print("✅ SafeWhisperActor test passed")

            # Test cleanup
            cleanup_result = ray.get(actor.cleanup.remote())
            print(f"   Cleanup: {cleanup_result}")

            return True
        else:
            print("❌ Model loading failed")
            return False

    except Exception as e:
        print(f"❌ SafeWhisperActor test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    print("SafeWhisperActor - CUDA Double Free Fix")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test
        success = test_safe_whisper_actor()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python safe_whisper_actor.py test    # Run tests")
        print("")
        print("This actor provides:")
        print("  ✅ Safe CUDA handling (avoids double free)")
        print("  ✅ CPU-only Whisper (most stable)")
        print("  ✅ Proper memory management")
        print("  ✅ Conservative resource allocation")
