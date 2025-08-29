"""
src/services/transcription_service_simple.py

Simplified transcription service that works reliably.
Falls back to basic Ray tasks when Jobs cause issues.
"""

import asyncio
import uuid
import ray
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from src.models.videotools_model import (
    TranscriptionReqModel,
    TranscriptionURLReqModel,
    TaskRespModel,
    languageDetectionModel,
)
from src.models.pydantic_models import JSONModel, ASRModel, WordModel, SegmentModel
from src.database.mongodb import MongoDB
from src.utils.ray_client import RayClient


class SimpleTranscriptionService:
    """
    Simplified transcription service using Ray tasks instead of complex Jobs.
    More reliable and easier to debug.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task using Ray tasks."""
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

        # Process transcription asynchronously
        asyncio.create_task(self._process_transcription_with_ray_task(task_id, request))

        return task_id

    async def _process_transcription_with_ray_task(
        self, task_id: str, request: TranscriptionReqModel
    ):
        """Process transcription using Ray tasks (simpler than Jobs)."""
        try:
            # Update status to processing
            await self.db.update_task(task_id, {"status": "processing"})

            # Get file and create temp path
            file_data, filename = await self.db.get_file(request.file_id)
            if not file_data:
                await self.db.update_task(
                    task_id, {"status": "failed", "error_message": "File not found"}
                )
                return

            # Create temp file for processing
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"{task_id}_{filename}"

            with open(temp_path, "wb") as f:
                f.write(file_data)

            print(f"🚀 Starting Ray task for transcription: {task_id}")

            # Create Ray task for transcription
            @ray.remote(num_gpus=1 if request.gpu else 0, num_cpus=2, memory=4000000000)
            def transcription_task(audio_path: str, config: dict):
                try:
                    import sys

                    sys.path.append("/app/src")

                    from faster_whisper import WhisperModel
                    import torch

                    print(f"🎯 Processing audio: {audio_path}")
                    print(f"🤖 Model: {config['model']}")
                    print(f"🔧 GPU: {config['use_gpu']}")

                    # Initialize Whisper
                    device = (
                        "cuda"
                        if config["use_gpu"] and torch.cuda.is_available()
                        else "cpu"
                    )
                    compute_type = "float16" if device == "cuda" else "int8"

                    model = WhisperModel(
                        config["model"],
                        device=device,
                        compute_type=compute_type,
                        download_root="/app/models/whisper",
                    )

                    print(f"✅ Whisper model loaded on {device}")

                    # Transcribe with word timestamps
                    segments, info = model.transcribe(
                        audio_path,
                        language=config.get("language"),
                        initial_prompt=config.get("initial_prompt"),
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )

                    # Convert to our format
                    result_segments = []
                    words = []
                    full_text = ""

                    for segment in segments:
                        segment_words = []
                        segment_text = segment.text.strip()
                        full_text += segment_text + " "

                        if hasattr(segment, "words") and segment.words:
                            for word in segment.words:
                                word_dict = {
                                    "start": word.start,
                                    "end": word.end,
                                    "text": word.word.strip(),
                                    "confidence": getattr(word, "probability", None),
                                }
                                words.append(word_dict)
                                segment_words.append(word_dict)

                        result_segments.append(
                            {
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment_text,
                                "words": segment_words,
                            }
                        )

                    result = {
                        "segments": result_segments,
                        "words": words,
                        "text": full_text.strip(),
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": sum(s["end"] - s["start"] for s in result_segments),
                        "model": config["model"],
                    }

                    print(
                        f"✅ Transcription completed: {len(words)} words, {len(result_segments)} segments"
                    )
                    return result

                except Exception as e:
                    print(f"❌ Ray task failed: {e}")
                    import traceback

                    traceback.print_exc()
                    raise

            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

            # Submit Ray task
            task_config = {
                "model": request.model,
                "language": request.language if request.language != "auto" else None,
                "initial_prompt": request.prompt,
                "use_gpu": request.gpu,
            }

            future = transcription_task.remote(str(temp_path), task_config)

            # Get result (this will block until completed)
            result = ray.get(future)

            print(f"✅ Ray task completed for {task_id}")

            # Store results in database
            await self._store_transcription_results(task_id, result, request)

            # Update task status
            await self.db.update_task(
                task_id,
                {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "result_summary": {
                        "language": result.get("language"),
                        "duration": result.get("duration", 0),
                        "words_count": len(result.get("words", [])),
                        "segments_count": len(result.get("segments", [])),
                    },
                },
            )

            print(f"🎉 Transcription completed successfully: {task_id}")

            # Cleanup temp file
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp file: {e}")

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
            print(f"❌ Transcription processing failed for task {task_id}: {error_msg}")

    async def _store_transcription_results(
        self, task_id: str, ray_result: dict, request: TranscriptionReqModel
    ):
        """Store transcription results in database."""
        from src.utils.subtitle_formats import SubtitleFormatter

        try:
            # Convert to our models
            words = [
                WordModel(
                    start=word["start"],
                    end=word["end"],
                    text=word["text"],
                    confidence=word.get("confidence"),
                    speaker=word.get("speaker"),
                )
                for word in ray_result.get("words", [])
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
                            speaker=word.get("speaker"),
                        )
                        for word in segment.get("words", [])
                    ],
                )
                for segment in ray_result.get("segments", [])
            ]

            json_result = JSONModel(
                text=ray_result.get("text", ""),
                segments=segments,
                language=ray_result.get("language"),
                language_probability=ray_result.get("language_probability"),
            )

            # Generate subtitle formats
            subtitle_formatter = SubtitleFormatter()
            srt_content = subtitle_formatter.to_srt(segments)
            vtt_content = subtitle_formatter.to_vtt(segments)
            txt_content = ray_result.get("text", "")

            # Store results
            result_data = {
                "json_result": json_result.dict(),
                "srt_content": srt_content,
                "vtt_content": vtt_content,
                "txt_content": txt_content,
                "ray_result": ray_result,
            }

            await self.db.store_result(task_id, result_data)

        except Exception as e:
            print(f"❌ Error storing results for task {task_id}: {e}")
            raise

    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using simple Ray task."""
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

            @ray.remote(num_cpus=1, memory=2000000000)
            def language_detection_task(audio_path: str):
                try:
                    from faster_whisper import WhisperModel

                    # Use CPU to avoid GPU conflicts
                    model = WhisperModel(
                        "base",
                        device="cpu",
                        compute_type="int8",
                        download_root="/app/models/whisper",
                    )

                    # Quick language detection
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
                    print(f"Language detection failed: {e}")
                    return {"language": "en", "confidence": 0.5}

            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

            # Submit and get result
            future = language_detection_task.remote(str(temp_path))
            result = ray.get(future)

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
            return languageDetectionModel(
                file_id=file_id, language="en", confidence=0.5
            )

    # Task management methods (same as before)
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
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )
        return success

    # Result methods (same as hybrid version)
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
        """Get SRT result."""
        result = await self.db.get_result(task_id)
        if not result:
            return None
        return result["result_data"].get("srt_content")

    async def get_vtt_result(self, task_id: str) -> Optional[str]:
        """Get VTT result."""
        result = await self.db.get_result(task_id)
        if not result:
            return None
        return result["result_data"].get("vtt_content")

    async def get_txt_result(self, task_id: str) -> Optional[str]:
        """Get TXT result."""
        result = await self.db.get_result(task_id)
        if not result:
            return None
        return result["result_data"].get("txt_content")

    # URL transcription (simplified)
    async def start_transcription_from_url(
        self, request: TranscriptionURLReqModel
    ) -> str:
        """Start transcription from URL (simplified)."""
        # For now, return error - implement if needed
        raise Exception("URL transcription not implemented in simple version")


# For backward compatibility
TranscriptionService = SimpleTranscriptionService
