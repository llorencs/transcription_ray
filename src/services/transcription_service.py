"""
src/services/transcription_service.py

Hybrid transcription service that uses Ray Jobs + Actors for optimal performance
with visible driver logs.
"""

import asyncio
import uuid
import ray
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
        from ray.job_submission import JobSubmissionClient

        try:
            # Connect to Ray cluster
            client = JobSubmissionClient("http://ray-head:8265")

            # Job script that will run the transcription
            job_script = f"""
import sys
import ray
from pathlib import Path

# Add src to path
sys.path.append('/app/src')

# Import required modules
from src.services.ray_actors import TranscriptionCoordinator
from src.database.mongodb import MongoDB
import asyncio
import json
from datetime import datetime

async def run_transcription():
    # Configuration from job
    config = {json.dumps(config)}
    
    print(f"🎯 Ray Job started for task: {{config['task_id']}}")
    print(f"📁 Audio path: {{config['audio_path']}}")
    print(f"🤖 Model: {{config['model']}}")
    print(f"🗣️ Diarization: {{config['diarize']}}")
    
    try:
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            
        print("✅ Ray initialized in job")
        
        # Create transcription coordinator
        coordinator = TranscriptionCoordinator.remote()
        print("✅ Transcription coordinator created")
        
        # Process transcription using Ray Actors
        result = await coordinator.process_transcription.remote(
            audio_path=config['audio_path'],
            model_size=config['model'],
            language=config['language'],
            initial_prompt=config['initial_prompt'],
            diarize=config['diarize'],
            use_gpu=config['use_gpu']
        )
        
        result = ray.get(result)
        print("✅ Transcription processing completed")
        
        # Connect to MongoDB to store results
        db = MongoDB("mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin")
        await db.connect()
        print("✅ Connected to MongoDB")
        
        # Store results in database
        await store_transcription_results(db, config['task_id'], result)
        print("✅ Results stored in database")
        
        # Update task status
        await db.update_task(config['task_id'], {{
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "result_summary": {{
                "language": result.get("transcription", {{}}).get("language"),
                "duration": result.get("processing_info", {{}}).get("total_duration", 0),
                "words_count": len(result.get("transcription", {{}}).get("words", [])),
                "segments_count": len(result.get("transcription", {{}}).get("segments", []))
            }}
        }})
        
        print(f"🎉 Task {{config['task_id']}} completed successfully!")
        
        # Cleanup temp file
        try:
            Path(config['audio_path']).unlink(missing_ok=True)
            print("✅ Temp file cleaned up")
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp file: {{e}}")
            
    except Exception as e:
        print(f"❌ Ray Job failed: {{e}}")
        import traceback
        traceback.print_exc()
        
        # Update task status to failed
        try:
            db = MongoDB("mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin")
            await db.connect()
            await db.update_task(config['task_id'], {{
                "status": "failed",
                "error_message": str(e),
                "failed_at": datetime.utcnow()
            }})
        except Exception as db_error:
            print(f"❌ Failed to update task status: {{db_error}}")
        
        raise

async def store_transcription_results(db, task_id, ray_result):
    \"\"\"Store transcription results in database.\"\"\"
    from src.utils.subtitle_formats import SubtitleFormatter
    from src.models.pydantic_models import WordModel, SegmentModel, JSONModel, ASRModel
    
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
"""

            # Submit job
            job_id = client.submit_job(
                entrypoint='python -c "' + job_script.replace('"', '\\"') + '"',
                runtime_env={
                    "working_dir": "/app",
                    "pip": ["faster-whisper", "pyannote.audio", "librosa"],
                },
            )

            print(f"✅ Ray Job submitted: {job_id}")
            return job_id

        except Exception as e:
            print(f"❌ Failed to submit Ray Job: {e}")
            raise

    async def _monitor_ray_job(
        self, task_id: str, ray_job_id: str, request: TranscriptionReqModel
    ):
        """Monitor Ray Job status and handle completion."""
        from ray.job_submission import JobSubmissionClient

        try:
            client = JobSubmissionClient("http://ray-head:8265")

            print(f"📊 Monitoring Ray Job {ray_job_id} for task {task_id}")

            # Poll job status
            while True:
                status = client.get_job_status(ray_job_id)
                print(f"📈 Job {ray_job_id} status: {status}")

                if status.is_terminal():
                    if status == "SUCCEEDED":
                        print(f"✅ Ray Job {ray_job_id} completed successfully")
                        # Job should have updated the task status already
                        break
                    else:
                        # Job failed
                        logs = client.get_job_logs(ray_job_id)
                        error_msg = (
                            f"Ray Job failed with status: {status}\nLogs:\n{logs}"
                        )

                        await self.db.update_task(
                            task_id,
                            {
                                "status": "failed",
                                "error_message": error_msg,
                                "failed_at": datetime.utcnow(),
                            },
                        )
                        print(f"❌ Ray Job {ray_job_id} failed: {error_msg}")
                        break

                await asyncio.sleep(10)  # Check every 10 seconds

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

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = await self.db.get_task(task_id)
        if not task:
            return False

        # Cancel Ray Job if it exists
        if "ray_job_id" in task:
            try:
                from ray.job_submission import JobSubmissionClient

                client = JobSubmissionClient("http://ray-head:8265")
                client.stop_job(task["ray_job_id"])
                print(f"✅ Cancelled Ray Job: {task['ray_job_id']}")
            except Exception as e:
                print(f"⚠️ Failed to cancel Ray Job: {e}")

        # Update task status
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )
        return success

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
        """Detect language using Ray Job."""
        try:
            # For language detection, we'll use a simpler Ray task instead of full job
            # since it's quick and doesn't need persistent logs

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

            # Submit Ray task for language detection
            @ray.remote
            def detect_language_task(audio_path: str):
                from src.services.ray_actors import WhisperTranscriptionActor

                actor = WhisperTranscriptionActor.remote("base", "auto")
                result = ray.get(actor.detect_language.remote(audio_path))
                return result

            if not ray.is_initialized():
                ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

            result = ray.get(detect_language_task.remote(str(temp_path)))

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
            raise Exception(f"Language detection error: {str(e)}")


# For backward compatibility
TranscriptionService = HybridTranscriptionService
