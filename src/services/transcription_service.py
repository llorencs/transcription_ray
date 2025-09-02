#!/usr/bin/env python3
"""
Fixed Transcription Service using SafeWhisperActor.
"""

import ray
import os
import sys
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
sys.path.append("/app/src")

# Import the safe actor
sys.path.append("/app/scripts")
from safe_whisper_actor import SafeWhisperActor


class TranscriptionService:
    """
    Transcription service using SafeWhisperActor to avoid CUDA issues.
    """

    def __init__(self, ray_address: Optional[str] = None):
        """Initialize the service."""

        self.ray_address = ray_address or "ray://ray-head:10001"
        self.actors: Dict[str, Any] = {}  # actor_id -> actor_ref
        self.actor_pool_size = 2  # Conservative pool size
        self.default_model_size = "base"

        print(f"TranscriptionService initializing (Ray: {self.ray_address})")

    async def initialize(self) -> bool:
        """Initialize Ray connection and actor pool."""

        try:
            # Connect to Ray cluster
            if not ray.is_initialized():
                print("Connecting to Ray cluster...")
                ray.init(address=self.ray_address, ignore_reinit_error=True)
                print("✅ Connected to Ray cluster")

            # Create actor pool
            print(f"Creating {self.actor_pool_size} SafeWhisperActors...")
            for i in range(self.actor_pool_size):
                actor_id = f"whisper_actor_{i}"
                actor = SafeWhisperActor.remote()
                self.actors[actor_id] = actor
                print(f"   Created {actor_id}")

            # Load models in all actors
            print("Loading Whisper models...")
            load_tasks = []
            for actor_id, actor in self.actors.items():
                task = actor.load_model.remote(self.default_model_size, force_cpu=True)
                load_tasks.append((actor_id, task))

            # Wait for all models to load
            all_loaded = True
            for actor_id, task in load_tasks:
                try:
                    result = ray.get(task, timeout=120)  # 2 minutes timeout
                    if result["success"]:
                        print(f"   ✅ {actor_id}: Model loaded")
                    else:
                        print(f"   ❌ {actor_id}: Load failed - {result.get('error')}")
                        all_loaded = False
                except Exception as e:
                    print(f"   ❌ {actor_id}: Exception - {e}")
                    all_loaded = False

            if not all_loaded:
                print("⚠️ Some actors failed to load models, but continuing...")

            print("✅ TranscriptionService initialized")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize TranscriptionService: {e}")
            traceback.print_exc()
            return False

    def get_available_actor(self) -> Optional[Any]:
        """Get an available actor from the pool."""

        # For now, use round-robin selection
        # In production, you might want to check actor availability
        if not self.actors:
            return None

        actor_ids = list(self.actors.keys())
        # Simple selection - could be improved with load balancing
        selected_id = actor_ids[0]
        return self.actors[selected_id]

    async def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Target language (None for auto-detect)
            task: 'transcribe' or 'translate'
            **kwargs: Additional parameters
        """

        try:
            # Get available actor
            actor = self.get_available_actor()
            if not actor:
                return {
                    "success": False,
                    "error": "No available actors",
                    "timestamp": datetime.now().isoformat(),
                }

            # Prepare transcription parameters
            transcribe_params = {
                "language": language,
                "task": task,
                "beam_size": kwargs.get("beam_size", 5),
                "vad_filter": kwargs.get("vad_filter", True),
            }

            print(f"Transcribing {audio_path} with params: {transcribe_params}")

            # Perform transcription
            task_future = actor.transcribe_file.remote(audio_path, **transcribe_params)
            result = ray.get(task_future, timeout=300)  # 5 minutes timeout

            # Add metadata
            result["timestamp"] = datetime.now().isoformat()
            result["service"] = "TranscriptionService"
            result["parameters"] = transcribe_params

            if result["success"]:
                print(
                    f"✅ Transcription completed: {len(result.get('segments', []))} segments"
                )
            else:
                print(f"❌ Transcription failed: {result.get('error')}")

            return result

        except Exception as e:
            error_msg = f"Transcription service error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()

            return {
                "success": False,
                "error": error_msg,
                "file": audio_path,
                "timestamp": datetime.now().isoformat(),
            }

    async def transcribe_with_diarization(
        self, audio_path: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe with speaker diarization.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
        """

        try:
            actor = self.get_available_actor()
            if not actor:
                return {"success": False, "error": "No available actors"}

            task_future = actor.transcribe_with_diarization.remote(audio_path, **kwargs)
            result = ray.get(task_future, timeout=600)  # 10 minutes timeout

            result["timestamp"] = datetime.now().isoformat()
            result["service"] = "TranscriptionService"

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Diarization error: {str(e)}",
                "file": audio_path,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status and actor health."""

        status = {
            "service": "TranscriptionService",
            "ray_initialized": ray.is_initialized(),
            "ray_address": self.ray_address,
            "actor_count": len(self.actors),
            "timestamp": datetime.now().isoformat(),
        }

        # Check actor status
        actor_status = {}
        for actor_id, actor in self.actors.items():
            try:
                info_task = actor.get_system_info.remote()
                info = ray.get(info_task, timeout=10)
                actor_status[actor_id] = {"healthy": True, "info": info}
            except Exception as e:
                actor_status[actor_id] = {"healthy": False, "error": str(e)}

        status["actors"] = actor_status

        # Ray cluster info
        if ray.is_initialized():
            try:
                status["cluster_resources"] = ray.cluster_resources()
                status["available_resources"] = ray.available_resources()
            except Exception as e:
                status["cluster_error"] = str(e)

        return status

    async def cleanup(self) -> bool:
        """Clean up service resources."""

        try:
            print("Cleaning up TranscriptionService...")

            # Cleanup actors
            cleanup_tasks = []
            for actor_id, actor in self.actors.items():
                task = actor.cleanup.remote()
                cleanup_tasks.append((actor_id, task))

            # Wait for cleanup
            for actor_id, task in cleanup_tasks:
                try:
                    ray.get(task, timeout=30)
                    print(f"   ✅ {actor_id}: Cleaned up")
                except Exception as e:
                    print(f"   ❌ {actor_id}: Cleanup failed - {e}")

            self.actors.clear()

            print("✅ TranscriptionService cleanup completed")
            return True

        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False


# Test function
async def test_transcription_service():
    """Test the TranscriptionService."""

    print("🧪 Testing TranscriptionService...")

    service = TranscriptionService()

    try:
        # Initialize service
        if not await service.initialize():
            print("❌ Service initialization failed")
            return False

        # Get status
        status = await service.get_service_status()
        print(f"Service status: {status}")

        # Test with dummy file (if available)
        test_files = [
            "/app/temp/test.wav",
            "/app/temp/sample.mp3",
            "/app/samples/test.wav",
        ]

        test_file = None
        for file_path in test_files:
            if Path(file_path).exists():
                test_file = file_path
                break

        if test_file:
            print(f"Testing transcription with: {test_file}")
            result = await service.transcribe_file(test_file)
            print(f"Transcription result: {result}")
        else:
            print("No test audio files found - skipping transcription test")

        print("✅ TranscriptionService test completed")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

    finally:
        await service.cleanup()


if __name__ == "__main__":
    import asyncio

    print("TranscriptionService - Safe CUDA Version")
    print("=" * 50)

    async def main():
        success = await test_transcription_service()
        return 0 if success else 1

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
