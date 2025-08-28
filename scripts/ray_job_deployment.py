#!/usr/bin/env python3
"""
scripts/ray_job_deployment.py

Ray Job submission script for transcription service.
This allows us to see driver logs and better monitor the deployment.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ray import serve
import ray


def deploy_transcription_service():
    """Deploy the transcription service using Ray Serve within a Ray Job."""

    print("🚀 Starting transcription service deployment...")

    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            print("✅ Connected to Ray cluster")

        # Start Ray Serve
        try:
            serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
            print("✅ Ray Serve started")
        except Exception as e:
            print(f"⚠️  Ray Serve already running or failed to start: {e}")

        # Deploy from config file
        config_path = "/app/config/serve_config.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"📄 Deploying from config: {config_path}")
        serve.run(target=config_path, host="0.0.0.0", port=8000)

        print("🎉 Transcription service deployed successfully!")
        print("📊 Service endpoints:")
        print("  • Health: http://localhost:8000/health")
        print("  • Transcribe: http://localhost:8000/transcribe")
        print("  • Language Detection: http://localhost:8000/detect-language")
        print("  • Ray Dashboard: http://localhost:8265")

        # Keep the job running
        print("🔄 Service is running... Press Ctrl+C to stop")

        # Wait indefinitely (job will keep running until stopped)
        import time

        while True:
            try:
                # Check service status periodically
                applications = serve.status().applications
                print(f"📈 Active applications: {list(applications.keys())}")

                for app_name, app_status in applications.items():
                    print(f"  • {app_name}: {app_status.status}")

                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                print("🛑 Stopping service...")
                break
            except Exception as e:
                print(f"❌ Error checking status: {e}")
                time.sleep(10)

    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        print("🧹 Cleaning up...")


if __name__ == "__main__":
    deploy_transcription_service()
