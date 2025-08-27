"""
Ray client for distributed task management.
"""

import os
import ray
from typing import Dict, Any, Optional
import asyncio
from ray import serve


class RayClient:
    def __init__(self):
        self.ray_address = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
        self.connected = False

    async def connect(self):
        """Connect to Ray cluster."""
        try:
            if not ray.is_initialized():
                ray.init(address=self.ray_address, ignore_reinit_error=True)

            self.connected = True
            print("Connected to Ray cluster successfully")

            # Initialize Ray Serve
            try:
                serve.start(
                    detached=True, http_options={"host": "0.0.0.0", "port": 8000}
                )
            except Exception as e:
                print(f"Ray Serve already started or failed to start: {e}")

        except Exception as e:
            print(f"Failed to connect to Ray: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Ray cluster."""
        if self.connected and ray.is_initialized():
            ray.shutdown()
            self.connected = False

    def submit_task(self, task_func, *args, **kwargs):
        """Submit a task to Ray."""
        if not self.connected:
            raise RuntimeError("Ray client not connected")

        return task_func.remote(*args, **kwargs)

    def get_task_result(self, object_ref, timeout: Optional[float] = None):
        """Get result from Ray task."""
        if not self.connected:
            raise RuntimeError("Ray client not connected")

        return ray.get(object_ref, timeout=timeout)

    def get_task_status(self, object_ref):
        """Get status of Ray task."""
        if not self.connected:
            raise RuntimeError("Ray client not connected")

        try:
            ray.get(object_ref, timeout=0.001)
            return "completed"
        except ray.exceptions.GetTimeoutError:
            return "running"
        except Exception:
            return "failed"

    def cancel_task(self, object_ref):
        """Cancel a Ray task."""
        if not self.connected:
            raise RuntimeError("Ray client not connected")

        try:
            ray.cancel(object_ref)
            return True
        except Exception as e:
            print(f"Failed to cancel task: {e}")
            return False

    async def deploy_model(self, model_name: str, model_class, *args, **kwargs):
        """Deploy a model using Ray Serve."""
        try:
            deployment = serve.deployment(
                name=model_name, num_replicas=1, route_prefix=f"/{model_name}"
            )(model_class)

            deployment.deploy(*args, **kwargs)
            print(f"Model {model_name} deployed successfully")

        except Exception as e:
            print(f"Failed to deploy model {model_name}: {e}")
            raise

    def get_cluster_resources(self) -> Dict[str, Any]:
        """Get Ray cluster resource information."""
        if not self.connected:
            raise RuntimeError("Ray client not connected")

        return ray.cluster_resources()

    def get_cluster_nodes(self) -> Dict[str, Any]:
        """Get Ray cluster node information."""
        if not self.connected:
            raise RuntimeError("Ray client not connected")

        return ray.nodes()
