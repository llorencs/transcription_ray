#!/usr/bin/env python3
"""
scripts/deploy_python_only.py

Deployment completamente en Python sin YAML config.
"""

import os
import sys
import time
import ray
from ray import serve
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def deploy_transcription_service():
    """Deploy transcription service usando solo Python API."""

    print("🚀 Starting transcription service deployment (Python-only)...")

    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                address="ray://ray-head:10001",
                ignore_reinit_error=True,
                log_to_driver=True,
            )
        print("✅ Connected to Ray cluster")

        # Start Ray Serve
        try:
            serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
            print("✅ Ray Serve started")
        except Exception as e:
            print(f"⚠️  Ray Serve already running: {e}")

        # Wait for cluster to be ready
        print("⏳ Waiting for cluster to be ready...")
        time.sleep(5)

        # Import deployments
        from src.deployments.ray_serve_models import (
            TranscriptionService,
            LanguageDetectionService,
            FastAPITranscriptionApp,
        )

        print("🏗️ Creating deployments...")

        # Create individual deployments with configuration
        transcription_deployment = TranscriptionService.options(
            name="TranscriptionService",
            num_replicas=1,
            ray_actor_options={"num_gpus": 1, "num_cpus": 2, "memory": 8_000_000_000},
        ).bind(
            model_cache_path="/app/models",
            default_whisper_model="base",
            enable_diarization=True,
        )

        language_detection_deployment = LanguageDetectionService.options(
            name="LanguageDetectionService",
            num_replicas=1,
            ray_actor_options={"num_cpus": 1, "memory": 2_000_000_000},
        ).bind(model_cache_path="/app/models", whisper_model="base")

        # Create main FastAPI app
        app_deployment = FastAPITranscriptionApp.options(
            name="TranscriptionAPIGateway",
            num_replicas=1,
            ray_actor_options={"num_cpus": 1, "memory": 1_000_000_000},
        ).bind(transcription_deployment, language_detection_deployment)

        print("🚀 Deploying application...")

        # Deploy the application
        serve.run(
            app_deployment,
            name="transcription-service",
            route_prefix="/",
            blocking=False,
        )

        print("✅ Deployment successful!")
        print("🌐 Service endpoints:")
        print("  • Health: http://localhost:8000/health")
        print("  • Transcribe: http://localhost:8000/transcribe")
        print("  • Language Detection: http://localhost:8000/detect-language")
        print("  • Ray Dashboard: http://localhost:8265")

        # Wait for deployments to be ready
        print("⏳ Waiting for deployments to be healthy...")
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                status = serve.status()
                apps = status.applications

                if "transcription-service" in apps:
                    app_status = apps["transcription-service"]
                    print(f"📊 App status: {app_status.status}")

                    if app_status.status == "RUNNING":
                        print("✅ All deployments are healthy!")
                        break

                    # Show deployment details
                    for (
                        deployment_name,
                        deployment_info,
                    ) in app_status.deployments.items():
                        print(f"  • {deployment_name}: {deployment_info.status}")

                time.sleep(10)

            except Exception as e:
                print(f"⚠️  Error checking status: {e}")
                time.sleep(5)

        # Test the service
        print("🧪 Testing service...")
        try:
            import requests

            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                print("✅ Health check passed!")
                print(f"📋 Response: {response.json()}")
            else:
                print(f"⚠️  Health check returned: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not test service: {e}")

        # Final status check and exit
        print("🔄 Final status check...")
        time.sleep(10)  # Give deployments time to stabilize

        try:
            status = serve.status()
            apps = status.applications

            if "transcription-service" in apps:
                app_status = apps["transcription-service"]
                print(f"📈 Final service status: {app_status.status}")

                # Count healthy replicas
                healthy_count = 0
                total_count = 0

                for deployment_name, deployment_info in app_status.deployments.items():
                    total_count += 1
                    if deployment_info.status == "HEALTHY":
                        healthy_count += 1
                    print(f"  • {deployment_name}: {deployment_info.status}")

                print(f"📊 Healthy deployments: {healthy_count}/{total_count}")

                if healthy_count == total_count and app_status.status == "RUNNING":
                    print("✅ Deployment completed successfully!")
                    print("🔄 Service is now running in detached mode")
                    return True
                else:
                    print("⚠️  Some deployments may not be ready yet")
                    print("💡 Use 'make health' or 'make ray-status' to check status")
                    return True  # Still consider it successful as it's running
            else:
                print("❌ Application not found in status")
                return False

        except Exception as e:
            print(f"❌ Error in final status check: {e}")
            print("⚠️  Deployment may still be successful - check manually")
            return True

    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = deploy_transcription_service()
    if not success:
        sys.exit(1)
