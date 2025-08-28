#!/usr/bin/env python3
"""
scripts/monitor_service.py

Script para monitorear continuamente el servicio Ray Serve.
"""

import os
import sys
import time
import ray
from ray import serve
from pathlib import Path


def monitor_transcription_service():
    """Monitor transcription service continuously."""

    print("📊 Starting service monitoring...")

    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

        print("✅ Connected to Ray cluster")
        print("📊 Monitoring transcription service... Press Ctrl+C to stop")
        print("=" * 60)

        while True:
            try:
                # Get service status
                status = serve.status()
                apps = status.applications

                print(f"\n⏰ Status check at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

                if "transcription-service" in apps:
                    app_status = apps["transcription-service"]
                    print(f"📈 Application: {app_status.status}")

                    # Show deployment details
                    healthy_count = 0
                    total_count = 0

                    for (
                        deployment_name,
                        deployment_info,
                    ) in app_status.deployments.items():
                        total_count += 1
                        status_icon = (
                            "✅" if deployment_info.status == "HEALTHY" else "⚠️"
                        )

                        if deployment_info.status == "HEALTHY":
                            healthy_count += 1

                        print(
                            f"  {status_icon} {deployment_name}: {deployment_info.status}"
                        )

                        # Show replica details if available
                        if hasattr(deployment_info, "replicas"):
                            for replica_info in deployment_info.replicas:
                                print(f"    └─ Replica: {replica_info.state}")

                    print(f"📊 Healthy: {healthy_count}/{total_count}")

                    # Test endpoints
                    try:
                        import requests

                        # Test health endpoint
                        response = requests.get(
                            "http://localhost:8000/health", timeout=5
                        )
                        if response.status_code == 200:
                            print("🟢 Health endpoint: OK")
                        else:
                            print(f"🔴 Health endpoint: {response.status_code}")

                    except Exception as e:
                        print(f"🔴 Health endpoint: Error - {e}")

                else:
                    print("❌ Transcription service not found")
                    print("💡 Available applications:", list(apps.keys()))

                # Ray cluster status
                try:
                    cluster_resources = ray.cluster_resources()
                    print(
                        f"🖥️  Cluster: {cluster_resources.get('CPU', 0):.1f} CPU, {cluster_resources.get('GPU', 0)} GPU"
                    )
                except Exception as e:
                    print(f"⚠️  Cluster status error: {e}")

                print("─" * 60)

                # Wait before next check
                time.sleep(30)

            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        return True

    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def quick_status_check():
    """Quick status check without continuous monitoring."""

    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

        print("📊 Quick Status Check")
        print("=" * 40)

        # Get service status
        status = serve.status()
        apps = status.applications

        if "transcription-service" in apps:
            app_status = apps["transcription-service"]
            print(f"📈 Application Status: {app_status.status}")

            for deployment_name, deployment_info in app_status.deployments.items():
                status_icon = "✅" if deployment_info.status == "HEALTHY" else "⚠️"
                print(f"  {status_icon} {deployment_name}: {deployment_info.status}")

            # Test health endpoint
            try:
                import requests

                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("🟢 Health Endpoint: Responding")
                    result = response.json()
                    print(f"   └─ {result}")
                else:
                    print(f"🔴 Health Endpoint: HTTP {response.status_code}")
            except Exception as e:
                print(f"🔴 Health Endpoint: {e}")

        else:
            print("❌ Transcription service not running")
            if apps:
                print("📋 Available applications:", list(apps.keys()))
            else:
                print("📋 No applications running")

        # Ray cluster info
        try:
            cluster_resources = ray.cluster_resources()
            print(f"🖥️  Cluster Resources:")
            print(f"   └─ CPU: {cluster_resources.get('CPU', 0):.1f}")
            print(f"   └─ GPU: {cluster_resources.get('GPU', 0)}")
            print(f"   └─ Memory: {cluster_resources.get('memory', 0) / 1e9:.1f} GB")
        except Exception as e:
            print(f"⚠️  Cluster status: {e}")

        return True

    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_status_check()
    else:
        success = monitor_transcription_service()

    if not success:
        sys.exit(1)
