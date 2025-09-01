#!/usr/bin/env python3
"""
Simple test script for the Direct Transcription Service
"""

import requests
import sys
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8080"


def test_direct_transcription(audio_file: str):
    """Test the direct transcription approach."""

    print("🧪 Testing Direct Transcription Service")
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

        # Step 4: Start transcription (basic settings)
        print("4. Starting transcription...")
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
        else:
            print(
                f"❌ Transcription start failed: {response.status_code} - {response.text}"
            )
            return False

        # Step 5: Monitor progress
        print("5. Monitoring transcription progress...")
        max_wait_time = 300  # 5 minutes
        check_interval = 5  # 5 seconds

        for i in range(max_wait_time // check_interval):
            response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")

            if response.status_code == 200:
                task_data = response.json()
                status = task_data["status"]

                print(
                    f"   Status: {status} (check {i+1}/{max_wait_time//check_interval})"
                )

                if status == "completed":
                    print("✅ Transcription completed successfully!")
                    break
                elif status == "failed":
                    error_msg = task_data.get("error_message", "Unknown error")
                    print(f"❌ Transcription failed: {error_msg}")
                    return False
                elif status in ["cancelled", "stopped"]:
                    print(f"🛑 Task was {status}")
                    return False

                time.sleep(check_interval)
            else:
                print(f"❌ Could not check task status: {response.status_code}")
                return False
        else:
            print("⏰ Transcription timed out")
            return False

        # Step 6: Download results
        print("6. Downloading results...")

        # Try to get JSON result
        response = requests.get(f"{API_BASE_URL}/results/{task_id}/json")
        if response.status_code == 200:
            json_result = response.json()
            text_preview = json_result.get("text", "")[:200]
            print(f"✅ JSON result: {len(json_result.get('segments', []))} segments")
            print(
                f"   Text preview: '{text_preview}{'...' if len(text_preview) >= 200 else ''}'"
            )
        else:
            print(f"⚠️ JSON result failed: {response.status_code}")

        # Try to get SRT result
        response = requests.get(f"{API_BASE_URL}/results/{task_id}/srt")
        if response.status_code == 200:
            srt_content = response.text
            print(f"✅ SRT result: {len(srt_content)} characters")
            # Save SRT file
            srt_file = Path(f"test_result_{task_id}.srt")
            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"   Saved to: {srt_file}")
        else:
            print(f"⚠️ SRT result failed: {response.status_code}")

        # Try to get TXT result
        response = requests.get(f"{API_BASE_URL}/results/{task_id}/txt")
        if response.status_code == 200:
            txt_content = response.text
            print(f"✅ TXT result: {len(txt_content)} characters")
            # Save TXT file
            txt_file = Path(f"test_result_{task_id}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(txt_content)
            print(f"   Saved to: {txt_file}")
        else:
            print(f"⚠️ TXT result failed: {response.status_code}")

        print("\n" + "=" * 60)
        print("🎉 Direct transcription test completed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_direct.py <audio_file>")
        print("Example: python scripts/test_direct.py /path/to/audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    success = test_direct_transcription(audio_file)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
