#!/usr/bin/env python3
"""
scripts/test_simple.py

Simple test to verify basic API functionality without Ray complications.
"""

import requests
import sys
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8080"


def test_basic_api():
    """Test basic API without transcription."""

    print("🧪 Testing basic API functionality...")

    try:
        # Test health check
        print("1. Testing health check...")
        response = requests.get(f"{API_BASE_URL}/health")

        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(
                f"   Ray Serve Available: {health_data.get('ray_serve_available', False)}"
            )
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

        # Test file upload
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            print(f"2. Testing file upload: {audio_file}")

            if not Path(audio_file).exists():
                print(f"❌ File not found: {audio_file}")
                return False

            with open(audio_file, "rb") as f:
                files = {"file": (Path(audio_file).name, f, "audio/mpeg")}
                response = requests.post(f"{API_BASE_URL}/files/upload", files=files)

            if response.status_code == 200:
                data = response.json()
                file_id = data["id"]
                print(f"✅ File uploaded: {file_id}")

                # Test file list
                print("3. Testing file list...")
                response = requests.get(f"{API_BASE_URL}/files")
                if response.status_code == 200:
                    files_data = response.json()
                    print(f"✅ Files listed: {len(files_data['files'])} files")
                else:
                    print(f"❌ File list failed: {response.status_code}")

                # Test language detection (simple)
                print("4. Testing language detection...")
                response = requests.post(
                    f"{API_BASE_URL}/detect-language", json={"file_id": file_id}
                )

                if response.status_code == 200:
                    lang_data = response.json()
                    print(
                        f"✅ Language detection: {lang_data.get('language')} (confidence: {lang_data.get('confidence', 'N/A')})"
                    )
                else:
                    print(
                        f"⚠️  Language detection failed: {response.status_code} - {response.text}"
                    )
                    # Don't fail the test for language detection

                return True
            else:
                print(
                    f"❌ File upload failed: {response.status_code} - {response.text}"
                )
                return False
        else:
            print("2. Skipping file upload (no file provided)")
            return True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False


def test_transcription_simple():
    """Test transcription with simple approach."""

    if len(sys.argv) < 2:
        print("⚠️  No audio file provided for transcription test")
        return True

    audio_file = sys.argv[1]
    print(f"🎯 Testing transcription: {audio_file}")

    try:
        # Upload file
        with open(audio_file, "rb") as f:
            files = {"file": (Path(audio_file).name, f, "audio/mpeg")}
            response = requests.post(f"{API_BASE_URL}/files/upload", files=files)

        if response.status_code != 200:
            print(f"❌ File upload failed: {response.status_code}")
            return False

        file_id = response.json()["id"]
        print(f"✅ File uploaded: {file_id}")

        # Start transcription
        transcription_request = {
            "file_id": file_id,
            "model": "base",
            "language": "auto",
            "diarize": False,
            "preprocess": False,
            "gpu": True,
            "asr_format": False,
        }

        response = requests.post(
            f"{API_BASE_URL}/transcribe", json=transcription_request
        )

        if response.status_code == 200:
            data = response.json()
            task_id = data["id"]
            print(f"✅ Transcription started: {task_id}")

            # Monitor task (simple polling)
            print("⏳ Monitoring task...")
            for i in range(60):  # Max 5 minutes
                response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")

                if response.status_code == 200:
                    task_data = response.json()
                    status = task_data["status"]
                    print(f"📊 Task status: {status}")

                    if status == "completed":
                        print("✅ Transcription completed!")

                        # Try to get results
                        response = requests.get(f"{API_BASE_URL}/results/{task_id}/txt")
                        if response.status_code == 200:
                            text_result = (
                                response.text[:200] + "..."
                                if len(response.text) > 200
                                else response.text
                            )
                            print(f"📝 Result preview: {text_result}")

                        return True

                    elif status == "failed":
                        error_msg = task_data.get("error_message", "Unknown error")
                        print(f"❌ Transcription failed: {error_msg}")
                        return False

                    elif status in ["cancelled", "stopped"]:
                        print(f"🛑 Task was {status}")
                        return False

                    time.sleep(5)
                else:
                    print(f"❌ Could not get task status: {response.status_code}")
                    return False

            print("⏰ Task timed out after 5 minutes")
            return False

        else:
            print(
                f"❌ Transcription start failed: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Simple API Test")
    print("=" * 50)

    # Test basic API first
    if not test_basic_api():
        print("❌ Basic API tests failed")
        sys.exit(1)

    print("\n" + "=" * 50)

    # Test transcription if file provided
    if not test_transcription_simple():
        print("❌ Transcription test failed")
        sys.exit(1)

    print("\n🎉 All tests passed!")
