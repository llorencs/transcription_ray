"""
API-focused transcription service that delegates heavy ML work to Ray workers.
This service doesn't import torch or other heavy ML dependencies.
"""

import uuid
import asyncio
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime
import tempfile
from pathlib import Path
import ray
import json

from src.models.videotools_model import (
    TranscriptionReqModel,
    TranscriptionURLReqModel,
    TaskRespModel,
    languageDetectionModel,
)
from src.models.pydantic_models import JSONModel, ASRModel
from src.database.mongodb import MongoDB
from src.utils.ray_client import RayClient


class APITranscriptionService:
    """Lightweight transcription service for the API layer."""

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task."""
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

        # Submit Ray task using Ray Serve
        try:
            # Call Ray Serve endpoint for transcription
            ray_task_id = await self._submit_transcription_to_ray(task_id, request)

            # Store Ray task reference
            await self.db.update_task(
                task_id, {"ray_task_id": str(ray_task_id), "status": "submitted"}
            )

        except Exception as e:
            await self.db.update_task(
                task_id,
                {
                    "status": "failed",
                    "error_message": f"Failed to submit to Ray: {str(e)}",
                },
            )
            raise

        return task_id

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

    async def _submit_transcription_to_ray(
        self, task_id: str, request: TranscriptionReqModel
    ) -> str:
        """Submit transcription task to Ray cluster."""
        # Get file path for processing
        file_data, filename = await self.db.get_file(request.file_id)
        if not file_data:
            raise FileNotFoundError(f"File {request.file_id} not found")

        # Create temp file path (Ray workers will handle the actual file)
        temp_dir = Path("/app/temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{request.file_id}_{filename}"

        with open(temp_path, "wb") as f:
            f.write(file_data)

        # Submit to Ray using remote function
        ray_task = self._process_transcription_remote.remote(
            task_id=task_id,
            audio_path=str(temp_path),
            model_size=request.model,
            language=request.language if request.language != "auto" else None,
            initial_prompt=request.prompt,
            diarize=request.diarize,
            use_gpu=request.gpu,
            preprocess=request.preprocess,
        )

        # Monitor task asynchronously
        asyncio.create_task(self._monitor_ray_task(task_id, ray_task, request))

        return str(ray_task)

    @ray.remote
    def _process_transcription_remote(
        self,
        task_id: str,
        audio_path: str,
        model_size: str = "base",
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        diarize: bool = False,
        use_gpu: bool = True,
        preprocess: bool = False,
    ) -> Dict[str, Any]:
        """Remote Ray function for transcription processing."""
        # Import heavy dependencies only in Ray workers
        from src.services.ray_actors import TranscriptionCoordinator

        # Initialize coordinator
        coordinator = TranscriptionCoordinator.remote()

        # Process transcription
        result = coordinator.process_transcription.remote(
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            initial_prompt=initial_prompt,
            diarize=diarize,
            use_gpu=use_gpu,
        )

        return ray.get(result)

    async def _monitor_ray_task(
        self, task_id: str, ray_task, request: TranscriptionReqModel
    ):
        """Monitor Ray task completion and store results."""
        try:
            # Update status to running
            await self.db.update_task(task_id, {"status": "running"})

            # Wait for task completion
            result = ray.get(ray_task)

            # Process and store results
            await self._store_transcription_results(task_id, result, request)

            # Update task status
            await self.db.update_task(
                task_id,
                {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "result_summary": {
                        "language": result["transcription"].get("language"),
                        "duration": result["processing_info"].get("total_duration"),
                        "words_count": len(result["transcription"]["words"]),
                        "segments_count": len(result["transcription"]["segments"]),
                    },
                },
            )

            # Call webhook if provided
            if request.callback_url:
                await self._call_webhook(
                    request.callback_url, task_id, "completed", result
                )

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

            # Call webhook for failure
            if request.callback_url:
                await self._call_webhook(
                    request.callback_url, task_id, "failed", {"error": error_msg}
                )

    async def _store_transcription_results(
        self, task_id: str, result: Dict, request: TranscriptionReqModel
    ):
        """Store transcription results in database."""
        from src.utils.subtitle_formats import SubtitleFormatter

        # Convert to our models
        json_result = self._convert_to_json_model(result)
        asr_result = self._convert_to_asr_model(result, task_id)

        # Generate subtitle formats
        subtitle_formatter = SubtitleFormatter()
        srt_content = subtitle_formatter.to_srt(json_result.segments)
        vtt_content = subtitle_formatter.to_vtt(json_result.segments)
        txt_content = json_result.text

        # Store results
        result_data = {
            "json_result": json_result.dict(),
            "asr_result": asr_result.dict(),
            "srt_content": srt_content,
            "vtt_content": vtt_content,
            "txt_content": txt_content,
            "processing_info": result.get("processing_info", {}),
            "diarization_info": result.get("diarization", {}),
        }

        await self.db.store_result(task_id, result_data)

    def _convert_to_json_model(self, result: Dict) -> JSONModel:
        """Convert Ray result to JSONModel."""
        from src.models.pydantic_models import WordModel, SegmentModel

        transcription = result["transcription"]

        # Convert words
        words = [
            WordModel(
                start=word["start"],
                end=word["end"],
                text=word["text"],
                confidence=word.get("confidence"),
                speaker=word.get("speaker"),
            )
            for word in transcription["words"]
        ]

        # Convert segments
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
                        speaker=word.get("speaker"),
                    )
                    for word in segment["words"]
                ],
            )
            for segment in transcription["segments"]
        ]

        return JSONModel(text=transcription["text"], segments=segments)

    def _convert_to_asr_model(self, result: Dict, task_id: str) -> ASRModel:
        """Convert Ray result to ASRModel using segmenter."""
        from src.models.pydantic_models import EventModel, WordModel
        from src.core.segmenter import Segmenter, SegmenterConfig

        transcription = result["transcription"]

        # Convert to WordModel objects for segmenter
        words = [
            WordModel(
                start=word["start"],
                end=word["end"],
                text=word["text"],
                confidence=word.get("confidence"),
                speaker=word.get("speaker"),
            )
            for word in transcription["words"]
        ]

        # Use segmenter to create events
        segmenter_config = SegmenterConfig()
        segmenter = Segmenter(segmenter_config)

        segments, events = segmenter.segment_words(words, transcription.get("language"))

        # Convert events to EventModel
        event_models = [
            EventModel(
                content=event.content,
                start_time=event.start_time,
                end_time=event.end_time,
                event_type=event.event_type,
                language=event.language,
                confidence=event.confidence,
                speaker=event.speaker,
            )
            for event in events
        ]

        return ASRModel(
            asr_model=result["processing_info"]["model_size"],
            created_at=result["processing_info"]["started_at"],
            generated_by="transcription-service",
            version=1,
            events=event_models,
        )

    async def _process_url_transcription(
        self, task_id: str, request: TranscriptionURLReqModel
    ):
        """Process transcription from URL."""
        try:
            # Update status
            await self.db.update_task(task_id, {"status": "downloading"})

            # Download file
            file_id = await self._download_file_from_url(request.url)

            # Update task with file_id
            await self.db.update_task(
                task_id, {"file_id": file_id, "status": "processing"}
            )

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

            # Submit transcription
            ray_task_id = await self._submit_transcription_to_ray(
                task_id, transcription_request
            )

            await self.db.update_task(
                task_id, {"ray_task_id": str(ray_task_id), "status": "running"}
            )

        except Exception as e:
            await self.db.update_task(
                task_id, {"status": "failed", "error_message": str(e)}
            )

    async def _download_file_from_url(self, url: str) -> str:
        """Download file from URL and store in database."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()

                # Get filename from URL or use default
                filename = Path(url).name or f"downloaded_{uuid.uuid4().hex[:8]}"

                # Store file
                file_id = await self.db.store_file(
                    response.content,
                    filename,
                    {"source_url": url, "downloaded_at": datetime.utcnow()},
                )

                return file_id

        except Exception as e:
            raise Exception(f"Failed to download file from URL: {str(e)}")

    async def _call_webhook(
        self, callback_url: str, task_id: str, status: str, data: Dict
    ):
        """Call webhook URL with task result."""
        try:
            webhook_data = {
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    callback_url, json=webhook_data, timeout=30.0
                )
                response.raise_for_status()

        except Exception as e:
            print(f"Webhook call failed for task {task_id}: {e}")

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

        if task["status"] not in ["pending", "running", "submitted"]:
            return False

        # Cancel Ray task if it exists
        if "ray_task_id" in task:
            try:
                ray_task_ref = ray.ObjectRef.from_hex(task["ray_task_id"])
                ray.cancel(ray_task_ref)
            except Exception as e:
                print(f"Failed to cancel Ray task: {e}")

        # Update task status
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )

        return success

    # Result retrieval methods
    async def get_json_result(self, task_id: str) -> Optional[JSONModel]:
        """Get JSON result for task."""
        result = await self.db.get_result(task_id)
        if not result or "json_result" not in result["result_data"]:
            return None

        return JSONModel(**result["result_data"]["json_result"])

    async def get_asr_result(self, task_id: str) -> Optional[ASRModel]:
        """Get ASR result for task."""
        result = await self.db.get_result(task_id)
        if not result or "asr_result" not in result["result_data"]:
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
        """Detect language of audio file."""
        # Submit language detection to Ray worker
        ray_task = self._detect_language_remote.remote(file_id)
        language_info = ray.get(ray_task)

        return languageDetectionModel(
            file_id=file_id,
            language=language_info["language"],
            confidence=language_info["confidence"],
        )

    @ray.remote
    def _detect_language_remote(self, file_id: str) -> Dict[str, Any]:
        """Remote function for language detection."""
        from src.services.ray_actors import WhisperTranscriptionActor

        # Get file and prepare for processing
        # This would be implemented similar to the main transcription
        # but only for language detection

        # For now, return a placeholder
        return {"language": "en", "confidence": 0.9}
