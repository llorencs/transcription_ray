#!/usr/bin/env python3
"""
scripts/deploy_hybrid.py

Simple deployment for hybrid approach.
No need for Ray Serve - just ensure Ray cluster is ready.
"""

import os
import sys
import time
import ray
from pathlib import Path


def deploy_hybrid_service():
    """Prepare Ray cluster for hybrid transcription service."""

    print("🚀 Preparing Ray cluster for hybrid transcription service...")

    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(
                address="ray://ray-head:10001",
                ignore_reinit_error=True,
                log_to_driver=True,
            )
        print("✅ Connected to Ray cluster")

        # Check cluster resources
        resources = ray.cluster_resources()
        print("📊 Cluster resources:")
        print(f"  • CPU: {resources.get('CPU', 0)}")
        print(f"  • GPU: {resources.get('GPU', 0)}")
        print(f"  • Memory: {resources.get('memory', 0) / 1e9:.1f} GB")

        # Test Ray Actors can be created
        print("🧪 Testing Ray Actor creation...")

        @ray.remote
        class TestActor:
            def ping(self):
                return "pong"

        test_actor = TestActor.remote()
        result = ray.get(test_actor.ping.remote())

        if result == "pong":
            print("✅ Ray Actors working correctly")
        else:
            raise Exception("Ray Actor test failed")

        # Check if we can import our transcription modules
        print("📦 Testing module imports...")

        try:
            # Add src to path
            sys.path.append("/app/src")
            from src.services.ray_actors import TranscriptionCoordinator

            print("✅ Ray transcription modules available")
        except ImportError as e:
            print(f"❌ Failed to import transcription modules: {e}")
            raise

        # Test MongoDB connection
        print("🗄️ Testing MongoDB connection...")
        try:
            from src.database.mongodb import MongoDB

            mongodb_url = "mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin"
            db = MongoDB(mongodb_url)
            # Note: We don't call connect() here as it's async
            print("✅ MongoDB module available")
        except ImportError as e:
            print(f"❌ Failed to import MongoDB module: {e}")
            raise

        print("🎉 Hybrid transcription service ready!")
        print("")
        print("📋 Service Architecture:")
        print("  • API FastAPI (port 8080) → Ray Jobs")
        print("  • Ray Jobs → Ray Actors (TranscriptionCoordinator)")
        print("  • Ray Actors → Whisper + Diarization models")
        print("  • Results → MongoDB")
        print("")
        print("✅ Benefits:")
        print("  • 🚀 Ray Actor performance (no HTTP overhead)")
        print("  • 📊 Driver logs visible in Ray Dashboard")
        print("  • 🔍 Job monitoring and management")
        print("  • 💾 Persistent model loading")
        print("")
        print("🌐 Endpoints available:")
        print("  • API: http://localhost:8080/docs")
        print("  • Ray Dashboard: http://localhost:8265")
        print("")
        print("🧪 Test with:")
        print("  make test TEST_AUDIO_FILE=/path/to/audio.wav")

        return True

    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = deploy_hybrid_service()
    if not success:
        sys.exit(1)
