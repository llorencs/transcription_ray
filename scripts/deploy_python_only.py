#!/usr/bin/env python3
"""
scripts/deploy_python_only.py

Deployment usando Ray Jobs para ver driver logs.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def deploy_as_ray_job():
    """Deploy usando Ray Job submission para tener driver logs visibles."""

    print("🚀 Deploying transcription service as Ray Job...")

    # Este script se ejecutará como un Ray Job
    # Los logs serán visibles en el Ray Dashboard

    try:
        import ray
        from ray import serve

        # Initialize Ray (ya debería estar inicializado como parte del job)
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001")

        print("✅ Ray initialized within job")

        # Start Ray Serve
        try:
            serve.start(detached=False)  # No detached para job
            print("✅ Ray Serve started within job")
        except Exception as e:
            print(f"⚠️  Ray Serve start: {e}")

        # Import deployments
        print("📦 Importing deployments...")
        from src.deployments.ray_serve_models import (
            TranscriptionService,
            LanguageDetectionService,
            FastAPITranscriptionApp,
        )

        print("🏗️ Creating deployments within job...")

        # Create deployments with explicit configuration
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

        # Create FastAPI app
        app_deployment = FastAPITranscriptionApp.options(
            name="TranscriptionAPIGateway",
            num_replicas=1,
            ray_actor_options={"num_cpus": 1, "memory": 1_000_000_000},
        ).bind(transcription_deployment, language_detection_deployment)

        print("🚀 Deploying application within job...")

        # Deploy the application
        serve.run(
            app_deployment,
            name="transcription-service",
            route_prefix="/",
            blocking=True,  # Block to keep job alive
            host="0.0.0.0",
            port=8000,
        )

        print("✅ Deployment successful within Ray Job!")

        # Keep job running
        print("🔄 Job running... Service deployed and accessible")

        # Monitor loop within job
        while True:
            try:
                print(f"📊 Job heartbeat: {time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Check service health
                status = serve.status()
                apps = status.applications

                if "transcription-service" in apps:
                    app_status = apps["transcription-service"]
                    print(f"📈 Service status: {app_status.status}")

                    # Test endpoint
                    try:
                        import requests

                        response = requests.get(
                            "http://localhost:8000/health", timeout=5
                        )
                        if response.status_code == 200:
                            print("🟢 Health check: OK")
                        else:
                            print(f"🔴 Health check: {response.status_code}")
                    except Exception as e:
                        print(f"🔴 Health check failed: {e}")

                else:
                    print("❌ Service not found in applications")

                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"❌ Error in monitoring: {e}")
                time.sleep(10)

    except Exception as e:
        print(f"❌ Job deployment failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    deploy_as_ray_job()
