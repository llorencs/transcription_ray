#!/usr/bin/env python3
"""
Simple Ray test to verify Ray Actors can access ML libraries.
"""

import ray
import sys


def test_ray_ml_access():
    """Test if Ray actors can access ML libraries."""

    print("🧪 Simple Ray ML Access Test")
    print("=" * 50)

    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            print("✅ Connected to Ray cluster")

        # Create a simple test actor
        @ray.remote
        def test_ml_imports():
            """Test function to run in Ray worker."""
            results = {}

            try:
                import torch

                results["torch"] = f"✅ SUCCESS - {torch.__version__}"
            except ImportError as e:
                results["torch"] = f"❌ FAILED - {str(e)}"

            try:
                import faster_whisper

                results["faster_whisper"] = "✅ SUCCESS"
            except ImportError as e:
                results["faster_whisper"] = f"❌ FAILED - {str(e)}"

            try:
                import librosa

                results["librosa"] = f"✅ SUCCESS - {librosa.__version__}"
            except ImportError as e:
                results["librosa"] = f"❌ FAILED - {str(e)}"

            try:
                import numpy

                results["numpy"] = f"✅ SUCCESS - {numpy.__version__}"
            except ImportError as e:
                results["numpy"] = f"❌ FAILED - {str(e)}"

            return results

        # Execute the test
        print("🔬 Testing ML imports in Ray worker...")
        future = test_ml_imports.remote()
        results = ray.get(future)

        print("\n📊 Results from Ray worker:")
        for lib, result in results.items():
            print(f"   {lib}: {result}")

        # Check if all imports succeeded
        all_success = all("✅ SUCCESS" in result for result in results.values())

        if all_success:
            print("\n✅ All ML libraries are accessible in Ray workers!")
            return True
        else:
            print("\n❌ Some ML libraries are missing in Ray workers!")
            return False

    except Exception as e:
        print(f"❌ Ray test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ray_ml_access()
    sys.exit(0 if success else 1)
