#!/usr/bin/env python3
"""
Test script for the Advanced Transcription Service API.
"""

import requests
import time
import json
from pathlib import Path
import argparse

API_BASE_URL = "http://localhost:8080"


def test_health_check():
    """Test API health check."""
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health check passed")


def upload_test_file(file_path: str):
    """Upload a test audio file."""
    print(f"Uploading test file: {file_path}")

    if not Path(file_path).exists():
        print(f"❌ Test file not found: {file_path}")
        return None

    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f, "audio/wav")}
        response = requests.post(f"{API_BASE_URL}/files/upload", files=files)

    if response.status_code == 200:
        data = response.json()
        file_id = data["id"]
        print(f"✓ File uploaded successfully: {file_id}")
        return file_id
    else:
        print(f"❌ File upload failed: {response.status_code} - {response.text}")
        return None


def start_transcription(
    file_id: str, model: str = "base", diarize: bool = False, preprocess: bool = False
):
    """Start a transcription task."""
    print(f"Starting transcription with model: {model}")

    payload = {
        "file_id": file_id,
        "model": model,
        "language": "auto",
        "diarize": diarize,
        "preprocess": preprocess,
        "gpu": True,
        "asr_format": True,
    }

    response = requests.post(f"{API_BASE_URL}/transcribe", json=payload)

    if response.status_code == 200:
        data = response.json()
        task_id = data["id"]
        print(f"✓ Transcription started: {task_id}")
        return task_id
    else:
        print(
            f"❌ Transcription failed to start: {response.status_code} - {response.text}"
        )
        return None


def wait_for_task_completion(task_id: str, timeout: int = 300):
    """Wait for task to complete."""
    print(f"Waiting for task completion: {task_id}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")

        if response.status_code == 200:
            data = response.json()
            status = data["status"]
            print(f"Task status: {status}")

            if status == "completed":
                print("✓ Task completed successfully")
                return True
            elif status == "failed":
                print(f"❌ Task failed: {data.get('error_message', 'Unknown error')}")
                return False
            elif status in ["cancelled"]:
                print(f"❌ Task was cancelled")
                return False

        time.sleep(5)

    print("❌ Task timed out")
    return False


def download_results(task_id: str, output_dir: str = "test_results"):
    """Download all result formats."""
    print(f"Downloading results for task: {task_id}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    formats = {"json": "json", "asr": "json", "srt": "srt", "vtt": "vtt", "txt": "txt"}

    for format_name, extension in formats.items():
        try:
            response = requests.get(f"{API_BASE_URL}/results/{task_id}/{format_name}")

            if response.status_code == 200:
                output_file = output_path / f"{task_id}.{extension}"

                if format_name in ["json", "asr"]:
                    # Save JSON with pretty formatting
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(response.json(), f, indent=2, ensure_ascii=False)
                else:
                    # Save text formats
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(response.text)

                print(f"✓ Downloaded {format_name}: {output_file}")
            else:
                print(f"❌ Failed to download {format_name}: {response.status_code}")

        except Exception as e:
            print(f"❌ Error downloading {format_name}: {e}")


def test_language_detection(file_id: str):
    """Test language detection."""
    print("Testing language detection...")

    response = requests.post(
        f"{API_BASE_URL}/detect-language", json={"file_id": file_id}
    )

    if response.status_code == 200:
        data = response.json()
        print(
            f"✓ Detected language: {data['language']} (confidence: {data.get('confidence', 'N/A')})"
        )
        return data
    else:
        print(f"❌ Language detection failed: {response.status_code} - {response.text}")
        return None


def test_file_management():
    """Test file listing and management."""
    print("Testing file management...")

    # List files
    response = requests.get(f"{API_BASE_URL}/files")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Listed {len(data['files'])} files")
        return data["files"]
    else:
        print(f"❌ Failed to list files: {response.status_code}")
        return []


def test_task_management():
    """Test task listing."""
    print("Testing task management...")

    response = requests.get(f"{API_BASE_URL}/tasks")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Listed {len(data['tasks'])} tasks")
        return data["tasks"]
    else:
        print(f"❌ Failed to list tasks: {response.status_code}")
        return []


def run_full_test(
    audio_file: str,
    model: str = "base",
    with_diarization: bool = False,
    with_preprocessing: bool = False,
):
    """Run complete API test."""
    print(f"\n{'='*60}")
    print(f"Running full API test")
    print(f"Audio file: {audio_file}")
    print(f"Model: {model}")
    print(f"Diarization: {with_diarization}")
    print(f"Preprocessing: {with_preprocessing}")
    print(f"{'='*60}\n")

    try:
        # Test 1: Health check
        test_health_check()

        # Test 2: Upload file
        file_id = upload_test_file(audio_file)
        if not file_id:
            return False

        # Test 3: Language detection
        test_language_detection(file_id)

        # Test 4: Start transcription
        task_id = start_transcription(
            file_id, model, with_diarization, with_preprocessing
        )
        if not task_id:
            return False

        # Test 5: Wait for completion
        if not wait_for_task_completion(task_id, timeout=600):
            return False

        # Test 6: Download results
        download_results(task_id)

        # Test 7: File and task management
        test_file_management()
        test_task_management()

        print(f"\n{'='*60}")
        print("✓ All tests passed successfully!")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Advanced Transcription Service API"
    )
    parser.add_argument("--audio-file", required=True, help="Path to test audio file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["base", "large-v3"],
        help="Whisper model to use",
    )
    parser.add_argument(
        "--diarization", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument(
        "--preprocessing", action="store_true", help="Enable audio preprocessing"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8080", help="API base URL"
    )

    args = parser.parse_args()

    global API_BASE_URL
    API_BASE_URL = args.api_url

    success = run_full_test(
        args.audio_file, args.model, args.diarization, args.preprocessing
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
