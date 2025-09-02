#!/usr/bin/env python3
"""
Test script for the Ray-based Transcription Service
"""

import requests
import sys
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8080"


def test_ray_transcription(audio_file: str):
    """Test the Ray-based transcription approach."""

    print("🧪 Testing Ray-based Transcription Service")
    print("=" * 60)

    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False

    try:
        # Step 1: Health check
        print("1. Testing health check...")
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed - Status: {health_data.get('status')}")
            print(
                f"   Ray Serve Available: {health_data.get('ray_serve_available', False)}"
            )
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

        # Step 2: Upload file
        print(f"2. Uploading audio file: {audio_file}")
        with open(audio_file, "rb") as f:
            files = {"file": (Path(audio_file).name, f, "audio/mpeg")}
            response = requests.post(f"{API_BASE_URL}/files/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            file_id = data["id"]
            print(f"✅ File uploaded successfully: {file_id}")
        else:
            print(f"❌ File upload failed: {response.status_code} - {response.text}")
            return False

        # Step 3: Test language detection
        print("3. Testing language detection...")
        response = requests.post(
            f"{API_BASE_URL}/detect-language", json={"file_id": file_id}
        )

        if response.status_code == 200:
            lang_data = response.json()
            print(
                f"✅ Language detected: {lang_data.get('language')} (confidence: {lang_data.get('confidence')})"
            )
        else:
            print(
                f"⚠️ Language detection failed: {response.status_code} - {response.text}"
            )

        # Step 4: Start Ray transcription
        print("4. Starting Ray-based transcription...")
        transcription_request = {
            "file_id": file_id,
            "model": "base",
            "language": "auto",
            "diarize": False,  # Start without diarization
            "preprocess": False,  # Start without preprocessing
            "gpu": True,
            "asr_format": False,
        }

        response = requests.post(
            f"{API_BASE_URL}/transcribe", json=transcription_request
        )

        if response.status_code == 200:
            data = response.json()
            task_id = data["id"]
            print(f"✅ Ray transcription started: {task_id}")
            print(f"   Message: {data.get('message', 'N/A')}")
        else:
            print(
                f"❌ Ray transcription start failed: {response.status_code} - {response.text}"
            )
            return False

        # Step 5: Monitor Ray processing
        print("5. Monitoring Ray transcription progress...")
        max_wait_time = 600  # 10 minutes for Ray processing
        check_interval = 10  # 10 seconds (Ray can be slower initially)

        for i in range(max_wait_time // check_interval):
            response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")

            if response.status_code == 200:
                task_data = response.json()
                status = task_data["status"]

                print(
                    f"   Ray Status: {status} (check {i+1}/{max_wait_time//check_interval})"
                )

                # Show result summary if available
                if "result" in task_data and task_data["result"]:
                    result_summary = task_data["result"]
                    if "model" in result_summary:
                        print(
                            f"     Model: {result_summary['model']}, GPU: {result_summary.get('gpu_used', 'N/A')}"
                        )

                if status == "completed":
                    print("✅ Ray transcription completed successfully!")
                    result_info = task_data.get("result", {})
                    print(f"   Language: {result_info.get('language', 'N/A')}")
                    print(f"   Duration: {result_info.get('duration', 'N/A')}s")
                    print(f"   Words: {result_info.get('words_count', 'N/A')}")
                    print(f"   Segments: {result_info.get('segments_count', 'N/A')}")
                    break
                elif status == "failed":
                    error_msg = task_data.get("error_message", "Unknown error")
                    print(f"❌ Ray transcription failed: {error_msg}")
                    return False
                elif status in ["cancelled", "stopped"]:
                    print(f"🛑 Ray task was {status}")
                    return False

                time.sleep(check_interval)
            else:
                print(f"❌ Could not check Ray task status: {response.status_code}")
                return False
        else:
            print("⏰ Ray transcription timed out after 10 minutes")
            return False

        # Step 6: Download Ray results
        print("6. Downloading Ray transcription results...")

        # Try to get JSON result
        response = requests.get(f"{API_BASE_URL}/results/{task_id}/json")
        if response.status_code == 200:
            json_result = response.json()
            text_preview = json_result.get("text", "")[:200]
            print(
                f"✅ Ray JSON result: {len(json_result.get('segments', []))} segments"
            )
            print(
                f"   Text preview: '{text_preview}{'...' if len(text_preview) >= 200 else ''}'"
            )
            print(f"   Language: {json_result.get('language', 'N/A')}")
        else:
            print(f"⚠️ Ray JSON result failed: {response.status_code}")

        # Try to get SRT result
        response = requests.get(f"{API_BASE_URL}/results/{task_id}/srt")
        if response.status_code == 200:
            srt_content = response.text
            print(f"✅ Ray SRT result: {len(srt_content)} characters")
            # Save SRT file
            srt_file = Path(f"ray_result_{task_id}.srt")
            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"   Saved to: {srt_file}")
        else:
            print(f"⚠️ Ray SRT result failed: {response.status_code}")

        # Try to get TXT result
        response = requests.get(f"{API_BASE_URL}/results/{task_id}/txt")
        if response.status_code == 200:
            txt_content = response.text
            print(f"✅ Ray TXT result: {len(txt_content)} characters")
            # Save TXT file
            txt_file = Path(f"ray_result_{task_id}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(txt_content)
            print(f"   Saved to: {txt_file}")
        else:
            print(f"⚠️ Ray TXT result failed: {response.status_code}")

        print("\n" + "=" * 60)
        print("🎉 Ray-based transcription test completed successfully!")
        print("🏗️ Architecture used:")
        print("   • API Container: FastAPI (lightweight)")
        print("   • Ray Workers: Whisper + ML models")
        print("   • Distributed processing with Ray Actors")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Ray test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_ray_cluster():
    """Check if Ray cluster is available."""
    try:
        import requests

        response = requests.get("http://localhost:8265/api/cluster_status", timeout=5)
        if response.status_code == 200:
            cluster_data = response.json()
            print(
                f"✅ Ray Cluster Status: {cluster_data.get('cluster_status', 'unknown')}"
            )
            return True
        else:
            print(f"⚠️ Ray cluster not responding: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Ray cluster check failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_ray.py <audio_file>")
        print("Example: python scripts/test_ray.py /path/to/audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    print("🔍 Pre-flight checks...")
    ray_available = check_ray_cluster()

    if not ray_available:
        print("⚠️ Ray cluster not available, but continuing test...")
        print("   (Ray will be initialized on demand)")

    print("")

    success = test_ray_transcription(audio_file)

    if success:
        print("\n✅ All Ray tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Ray tests failed!")
        sys.exit(1)
