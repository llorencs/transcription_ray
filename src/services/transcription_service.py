"""
src/services/transcription_service.py

Hybrid transcription service that uses Ray Jobs + Actors for optimal performance
with visible driver logs.
"""

import asyncio
import uuid
import ray
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.models.videotools_model import (
    TranscriptionReqModel,
    TranscriptionURLReqModel,
    TaskRespModel,
    languageDetectionModel,
)
from src.models.pydantic_models import JSONModel, ASRModel, WordModel, SegmentModel
from src.database.mongodb import MongoDB
from src.utils.ray_client import RayClient


class HybridTranscriptionService:
    """
    Hybrid service that submits Ray Jobs for transcription tasks.
    This gives us Ray Actor performance with visible driver logs.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task using Ray Job submission."""
        task_id = str(uuid.uuid4())

        # Create task record
        task_data = {
            "task_id": task_id,
            "file_id": request.file_id,
            "model": request.model,
            "gpu": request.gpu,
            "language": request.language,
            "prompt": request.prompt,
            "preprocess": request.preprocess,
            "diarize": request.diarize,
            "asr_format": request.asr_format,
            "callback_url": request.callback_url,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        await self.db.create_task(task_data)

        # Submit as Ray Job for visible logs
        asyncio.create_task(self._submit_transcription_job(task_id, request))

        return task_id

    async def _submit_transcription_job(
        self, task_id: str, request: TranscriptionReqModel
    ):
        """Submit transcription as Ray Job with actor-based processing."""
        try:
            # Update status to submitted
            await self.db.update_task(task_id, {"status": "submitted"})

            # Get file and prepare for processing
            file_data, filename = await self.db.get_file(request.file_id)
            if not file_data:
                await self.db.update_task(
                    task_id, {"status": "failed", "error_message": "File not found"}
                )
                return

            # Create temp file path for Ray Job
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"{task_id}_{filename}"

            with open(temp_path, "wb") as f:
                f.write(file_data)

            # Submit Ray Job
            job_config = {
                "task_id": task_id,
                "audio_path": str(temp_path),
                "model": request.model,
                "language": request.language if request.language != "auto" else None,
                "initial_prompt": request.prompt,
                "diarize": request.diarize,
                "use_gpu": request.gpu,
                "preprocess": request.preprocess,
            }

            print(f"🚀 Submitting Ray Job for task {task_id}")

            # Submit job using Ray Jobs API
            ray_job_id = self._submit_ray_job(job_config)

            # Store Ray job ID for tracking
            await self.db.update_task(
                task_id, {"ray_job_id": ray_job_id, "status": "running"}
            )

            # Monitor job completion asynchronously
            asyncio.create_task(self._monitor_ray_job(task_id, ray_job_id, request))

        except Exception as e:
            error_msg = str(e)
            await self.db.update_task(
                task_id,
                {
                    "status": "failed",
                    "error_message": error_msg,
                    "failed_at": datetime.utcnow(),
                },
            )
            print(f"❌ Job submission failed for task {task_id}: {error_msg}")

    def _submit_ray_job(self, config: Dict[str, Any]) -> str:
        """Submit Ray Job and return job ID."""
        import subprocess
        import json
        import tempfile

        try:
            # Create a proper Python script for the Ray Job
            job_script_content = f'''
import sys
import os
import ray
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append("/app/src")
sys.path.append("/app")

async def run_transcription():
    """Main transcription function running in Ray Job."""
    
    config = {json.dumps(config)}
    task_id = config["task_id"]
    
    print(f"🎯 Ray Job started for task: {{task_id}}")
    print(f"📁 Audio path: {{config['audio_path']}}")
    print(f"🤖 Model: {{config['model']}}")
    print(f"🗣️ Diarization: {{config['diarize']}}")
    
    try:
        # Initialize Ray - connect to existing cluster
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
        print("✅ Connected to Ray cluster from job")
        
        # Import after Ray initialization
        from src.services.ray_actors import TranscriptionCoordinator
        from src.database.mongodb import MongoDB
        
        print("✅ Imports successful")
        
        # Create transcription coordinator
        coordinator = TranscriptionCoordinator.remote()
        coordinator.initialize_actors.remote(config["model"], config["use_gpu"])
        print("✅ Transcription coordinator initialized")
        
        # Process transcription using Ray Actors
        result_ref = coordinator.process_transcription.remote(
            audio_path=config["audio_path"],
            model_size=config["model"],
            language=config.get("language"),
            initial_prompt=config.get("initial_prompt"),
            diarize=config["diarize"],
            use_gpu=config["use_gpu"]
        )
        
        print("⏳ Processing transcription...")
        result = ray.get(result_ref)
        print("✅ Transcription processing completed")
        
        # Connect to MongoDB to store results
        mongodb_url = "mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin"
        db = MongoDB(mongodb_url)
        await db.connect()
        print("✅ Connected to MongoDB")
        
        # Store results in database
        await store_transcription_results(db, task_id, result)
        print("✅ Results stored in database")
        
        # Update task status
        await db.update_task(task_id, {{
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "result_summary": {{
                "language": result.get("transcription", {{}}).get("language"),
                "duration": result.get("processing_info", {{}}).get("total_duration", 0),
                "words_count": len(result.get("transcription", {{}}).get("words", [])),
                "segments_count": len(result.get("transcription", {{}}).get("segments", []))
            }}
        }})
        
        print(f"🎉 Task {{task_id}} completed successfully!")
        
        # Cleanup temp file
        try:
            Path(config["audio_path"]).unlink(missing_ok=True)
            print("✅ Temp file cleaned up")
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp file: {{e}}")
            
        await db.disconnect()
            
    except Exception as e:
        print(f"❌ Ray Job failed: {{e}}")
        import traceback
        traceback.print_exc()
        
        # Update task status to failed
        try:
            from src.database.mongodb import MongoDB
            mongodb_url = "mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin"
            db = MongoDB(mongodb_url)
            await db.connect()
            await db.update_task(task_id, {{
                "status": "failed",
                "error_message": str(e),
                "failed_at": datetime.utcnow()
            }})
            await db.disconnect()
        except Exception as db_error:
            print(f"❌ Failed to update task status: {{db_error}}")
        
        raise

async def store_transcription_results(db, task_id, ray_result):
    """Store transcription results in database."""
    from src.utils.subtitle_formats import SubtitleFormatter
    from src.models.pydantic_models import WordModel, SegmentModel, JSONModel
    
    transcription = ray_result.get("transcription", {{}})
    
    # Convert to our models
    words = [
        WordModel(
            start=word["start"],
            end=word["end"], 
            text=word["text"],
            confidence=word.get("confidence"),
            speaker=word.get("speaker")
        )
        for word in transcription.get("words", [])
    ]
    
    segments = [
        SegmentModel(
            start=segment["start"],
            end=segment["end"],
            text=segment["text"],
            words=[
                WordModel(
                    start=word["start"],
                    end=word["end"],
                    text=word["text"], 
                    confidence=word.get("confidence"),
                    speaker=word.get("speaker")
                )
                for word in segment.get("words", [])
            ]
        )
        for segment in transcription.get("segments", [])
    ]
    
    json_result = JSONModel(
        text=transcription.get("text", ""),
        segments=segments,
        language=transcription.get("language"),
        language_probability=transcription.get("language_probability")
    )
    
    # Generate subtitle formats
    subtitle_formatter = SubtitleFormatter()
    srt_content = subtitle_formatter.to_srt(segments)
    vtt_content = subtitle_formatter.to_vtt(segments)
    txt_content = transcription.get("text", "")
    
    # Store results
    result_data = {{
        "json_result": json_result.dict(),
        "srt_content": srt_content,
        "vtt_content": vtt_content,
        "txt_content": txt_content,
        "ray_result": ray_result,  # Store complete result
    }}
    
    await db.store_result(task_id, result_data)

if __name__ == "__main__":
    asyncio.run(run_transcription())
'''

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(job_script_content)
                script_path = f.name

            # Submit job using ray job submit command - FIXED: Use --submission-id instead of --job-name
            submission_id = f"transcription-{config['task_id'][:8]}"
            cmd = [
                "ray",
                "job",
                "submit",
                "--address=ray://ray-head:10001",
                "--working-dir=/app",
                f"--submission-id={submission_id}",  # FIXED: Changed from --job-name to --submission-id
                "--",
                "python",
                script_path,
            ]

            print(f"🚀 Executing: {' '.join(cmd)}")

            # Execute ray job submit
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")

            if result.returncode == 0:
                # Extract job ID from output
                output_lines = result.stdout.strip().split("\n")
                for line in output_lines:
                    if "submitted successfully" in line:
                        print(f"✅ Ray Job submitted successfully")
                        # Use the submission_id as the job_id
                        return submission_id

                # Fallback - use submission_id
                print(f"✅ Ray Job submitted (using submission_id): {submission_id}")
                return submission_id
            else:
                error_msg = f"ray job submit failed: {result.stderr}"
                print(f"❌ {error_msg}")
                print(f"❌ stdout: {result.stdout}")
                raise Exception(error_msg)

        except Exception as e:
            print(f"❌ Failed to submit Ray Job: {e}")
            raise

    async def _monitor_ray_job(
        self, task_id: str, ray_job_id: str, request: TranscriptionReqModel
    ):
        """Monitor Ray Job status and handle completion."""
        import subprocess

        try:
            print(f"📊 Monitoring Ray Job {ray_job_id} for task {task_id}")

            # Poll job status using ray job status command
            while True:
                try:
                    # Get job status
                    result = subprocess.run(
                        [
                            "ray",
                            "job",
                            "status",
                            ray_job_id,
                            "--address=ray://ray-head:10001",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if result.returncode == 0:
                        status_output = result.stdout.strip()
                        print(f"📈 Job {ray_job_id} status output: {status_output}")

                        # Check if job is completed
                        if "SUCCEEDED" in status_output:
                            print(f"✅ Ray Job {ray_job_id} completed successfully")
                            # Job should have updated the task status already
                            break
                        elif "FAILED" in status_output:
                            # Get job logs
                            logs_result = subprocess.run(
                                [
                                    "ray",
                                    "job",
                                    "logs",
                                    ray_job_id,
                                    "--address=ray://ray-head:10001",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )

                            logs = (
                                logs_result.stdout
                                if logs_result.returncode == 0
                                else "Could not retrieve logs"
                            )
                            error_msg = f"Ray Job failed\nLogs:\n{logs}"

                            await self.db.update_task(
                                task_id,
                                {
                                    "status": "failed",
                                    "error_message": error_msg,
                                    "failed_at": datetime.utcnow(),
                                },
                            )
                            print(f"❌ Ray Job {ray_job_id} failed")
                            break
                        elif "STOPPED" in status_output:
                            await self.db.update_task(
                                task_id,
                                {
                                    "status": "cancelled",
                                    "cancelled_at": datetime.utcnow(),
                                },
                            )
                            print(f"🛑 Ray Job {ray_job_id} was stopped")
                            break
                    else:
                        print(f"⚠️ Could not get job status: {result.stderr}")

                except subprocess.TimeoutExpired:
                    print("⏰ Job status check timed out, continuing...")
                except Exception as status_error:
                    print(f"❌ Error checking job status: {status_error}")

                await asyncio.sleep(15)  # Check every 15 seconds

        except Exception as e:
            error_msg = f"Job monitoring failed: {str(e)}"
            await self.db.update_task(
                task_id,
                {
                    "status": "failed",
                    "error_message": error_msg,
                    "failed_at": datetime.utcnow(),
                },
            )
            print(f"❌ Job monitoring failed for task {task_id}: {error_msg}")

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = await self.db.get_task(task_id)
        if not task:
            return False

        # Cancel Ray Job if it exists
        if "ray_job_id" in task:
            try:
                result = subprocess.run(
                    [
                        "ray",
                        "job",
                        "stop",
                        task["ray_job_id"],
                        "--address=ray://ray-head:10001",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    print(f"✅ Cancelled Ray Job: {task['ray_job_id']}")
                else:
                    print(f"⚠️ Failed to cancel Ray Job: {result.stderr}")
            except Exception as e:
                print(f"⚠️ Failed to cancel Ray Job: {e}")

        # Update task status
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )
        return success

    async def start_transcription_from_url(
        self, request: TranscriptionURLReqModel
    ) -> str:
        """Start transcription task from URL."""
        task_id = str(uuid.uuid4())

        # Create task record
        task_data = {
            "task_id": task_id,
            "source_url": request.url,
            "model": request.model,
            "gpu": request.gpu,
            "language": request.language,
            "prompt": request.prompt,
            "preprocess": request.preprocess,
            "diarize": request.diarize,
            "asr_format": request.asr_format,
            "callback_url": request.callback_url,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        await self.db.create_task(task_data)

        # Process URL transcription asynchronously
        asyncio.create_task(self._process_url_transcription(task_id, request))

        return task_id

    async def _process_url_transcription(
        self, task_id: str, request: TranscriptionURLReqModel
    ):
        """Process transcription from URL."""
        try:
            import httpx

            # Update status
            await self.db.update_task(task_id, {"status": "downloading"})

            # Download file from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(request.url)
                response.raise_for_status()

                # Get filename from URL or use default
                filename = Path(request.url).name or f"downloaded_{task_id[:8]}"

                # Store file
                file_id = await self.db.store_file(
                    response.content,
                    filename,
                    {"source_url": request.url, "downloaded_at": datetime.utcnow()},
                )

            # Update task with file_id
            await self.db.update_task(task_id, {"file_id": file_id})

            # Create transcription request
            transcription_request = TranscriptionReqModel(
                file_id=file_id,
                model=request.model,
                gpu=request.gpu,
                language=request.language,
                prompt=request.prompt,
                preprocess=request.preprocess,
                diarize=request.diarize,
                asr_format=request.asr_format,
                callback_url=request.callback_url,
            )

            # Submit transcription job
            await self._submit_transcription_job(task_id, transcription_request)

        except Exception as e:
            await self.db.update_task(
                task_id, {"status": "failed", "error_message": str(e)}
            )

    # Task management methods
    async def get_task(self, task_id: str) -> Optional[TaskRespModel]:
        """Get task by ID."""
        task = await self.db.get_task(task_id)
        if not task:
            return None

        return TaskRespModel(
            id=task["task_id"],
            status=task["status"],
            result=task.get("result_summary"),
            error_message=task.get("error_message"),
        )

    async def list_tasks(self, skip: int = 0, limit: int = 100) -> List[TaskRespModel]:
        """List tasks with pagination."""
        tasks = await self.db.list_tasks(skip=skip, limit=limit)

        return [
            TaskRespModel(
                id=task["task_id"],
                status=task["status"],
                result=task.get("result_summary"),
                error_message=task.get("error_message"),
            )
            for task in tasks
        ]

    # Result retrieval methods (same as before)
    async def get_json_result(self, task_id: str) -> Optional[JSONModel]:
        """Get JSON result for task."""
        result = await self.db.get_result(task_id)
        if not result or "json_result" not in result["result_data"]:
            return None

        return JSONModel(**result["result_data"]["json_result"])

    async def get_asr_result(self, task_id: str) -> Optional[ASRModel]:
        """Get ASR result for task."""
        result = await self.db.get_result(task_id)
        if not result or not result["result_data"].get("asr_result"):
            return None

        return ASRModel(**result["result_data"]["asr_result"])

    async def get_srt_result(self, task_id: str) -> Optional[str]:
        """Get SRT subtitle result for task."""
        result = await self.db.get_result(task_id)
        if not result:
            return None
        return result["result_data"].get("srt_content")

    async def get_vtt_result(self, task_id: str) -> Optional[str]:
        """Get VTT subtitle result for task."""
        result = await self.db.get_result(task_id)
        if not result:
            return None
        return result["result_data"].get("vtt_content")

    async def get_txt_result(self, task_id: str) -> Optional[str]:
        """Get plain text result for task."""
        result = await self.db.get_result(task_id)
        if not result:
            return None
        return result["result_data"].get("txt_content")

    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using a simple Ray task (not actor)."""
        try:
            # Get file
            file_data, filename = await self.db.get_file(file_id)
            if not file_data:
                raise Exception("File not found")

            # Create temp file
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"lang_{file_id}_{filename}"

            with open(temp_path, "wb") as f:
                f.write(file_data)

            # Use a simple Ray task instead of actor to avoid memory issues
            @ray.remote
            def simple_language_detection(audio_path: str):
                try:
                    from faster_whisper import WhisperModel
                    import torch

                    device = (
                        "cpu"  # Use CPU for language detection to avoid GPU conflicts
                    )
                    compute_type = "int8"

                    model = WhisperModel(
                        "base",
                        device=device,
                        compute_type=compute_type,
                        download_root="/app/models/whisper",
                    )

                    # Just detect language, don't transcribe
                    segments, info = model.transcribe(
                        audio_path,
                        language=None,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )

                    # Consume first segment to trigger language detection
                    next(segments, None)

                    return {
                        "language": info.language,
                        "confidence": info.language_probability,
                    }

                except Exception as e:
                    print(f"Language detection task failed: {e}")
                    # Fallback to English
                    return {"language": "en", "confidence": 0.5}

            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

            # Submit task and get result
            result_ref = simple_language_detection.remote(str(temp_path))
            result = ray.get(result_ref)

            # Cleanup temp file
            try:
                temp_path.unlink(missing_ok=True)
            except:
                pass

            return languageDetectionModel(
                file_id=file_id,
                language=result["language"],
                confidence=result["confidence"],
            )

        except Exception as e:
            print(f"Language detection error: {e}")
            # Fallback result
            return languageDetectionModel(
                file_id=file_id, language="en", confidence=0.5
            )


# For backward compatibility
TranscriptionService = HybridTranscriptionService
