"""
src/services/transcription_service.py

Corrected Ray-based transcription service using Ray Actors directly.
No Docker commands, pure Ray architecture.
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
from src.models.pydantic_models import (
    JSONModel,
    ASRModel,
    WordModel,
    SegmentModel,
    EventModel,
)
from src.database.mongodb import MongoDB
from src.utils.ray_client import RayClient


class RayTranscriptionService:
    """
    Ray-based transcription service using Ray Actors directly.
    No subprocess calls, pure distributed processing.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task using Ray Actors."""
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

        # Process transcription asynchronously using Ray
        asyncio.create_task(self._process_transcription_with_ray(task_id, request))

        return task_id

    async def _process_transcription_with_ray(
        self, task_id: str, request: TranscriptionReqModel
    ):
        """Process transcription using Ray Actors."""
        temp_path = None

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

            print(f"🚀 Starting Ray Actor transcription: {task_id}")
            print(f"📁 Temp file: {temp_path} ({len(file_data)} bytes)")

            # Ensure Ray is initialized
            if not ray.is_initialized():
                try:
                    ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
                    print("✅ Connected to Ray cluster")
                except Exception as e:
                    print(f"❌ Ray connection failed: {e}")
                    # Fallback to local Ray for development
                    ray.init(ignore_reinit_error=True)
                    print("✅ Initialized local Ray")

            # Import and use Ray Actor
            from src.services.ray_actors import TranscriptionCoordinator

            # Create coordinator actor
            coordinator = TranscriptionCoordinator.remote()

            # Process transcription using Ray Actor
            print("🎯 Submitting to Ray TranscriptionCoordinator...")

            result_ref = coordinator.process_transcription.remote(
                audio_path=str(temp_path),
                model_size=request.model,
                language=request.language if request.language != "auto" else None,
                initial_prompt=request.prompt,
                diarize=request.diarize,  # Will be implemented later
                use_gpu=request.gpu,
                max_segment_duration=600.0,
            )

            # Get result with timeout
            try:
                result = ray.get(result_ref, timeout=600)  # 10 minute timeout
                print("✅ Ray Actor transcription completed")
            except ray.exceptions.RayTaskError as e:
                print(f"❌ Ray Actor error: {e}")
                raise Exception(f"Ray transcription failed: {str(e)}")
            except ray.exceptions.GetTimeoutError:
                print(f"❌ Ray Actor timeout after 10 minutes")
                raise Exception("Ray transcription timed out")

            if result:
                print(f"✅ Transcription completed successfully: {task_id}")

                # Store results in database
                await self._store_ray_results(task_id, result, request)

                # Update task status
                await self.db.update_task(
                    task_id,
                    {
                        "status": "completed",
                        "completed_at": datetime.utcnow(),
                        "result_summary": {
                            "language": result.get("transcription", {}).get("language"),
                            "duration": result.get("processing_info", {}).get(
                                "total_duration", 0
                            ),
                            "words_count": len(
                                result.get("transcription", {}).get("words", [])
                            ),
                            "segments_count": len(
                                result.get("transcription", {}).get("segments", [])
                            ),
                            "model": request.model,
                            "gpu_used": request.gpu,
                        },
                    },
                )
            else:
                raise Exception("Ray Actor returned empty result")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Ray transcription failed for task {task_id}: {error_msg}")
            import traceback

            traceback.print_exc()

            await self.db.update_task(
                task_id,
                {
                    "status": "failed",
                    "error_message": error_msg,
                    "failed_at": datetime.utcnow(),
                },
            )

        finally:
            # Cleanup temp files
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    print(f"🧹 Cleaned up temp audio file: {temp_path}")
                except Exception as e:
                    print(f"⚠️ Failed to cleanup temp audio file: {e}")

    async def _store_ray_results(
        self, task_id: str, ray_result: dict, request: TranscriptionReqModel
    ):
        """Store Ray transcription results in database."""
        from src.utils.subtitle_formats import SubtitleFormatter

        try:
            transcription = ray_result.get("transcription", {})

            # Convert to our models
            words = [
                WordModel(
                    start=word["start"],
                    end=word["end"],
                    text=word["text"],
                    confidence=word.get("confidence"),
                    speaker=word.get("speaker"),
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
                            speaker=word.get("speaker"),
                        )
                        for word in segment.get("words", [])
                    ],
                )
                for segment in transcription.get("segments", [])
            ]

            json_result = JSONModel(
                text=transcription.get("text", ""),
                segments=segments,
                language=transcription.get("language"),
                language_probability=transcription.get("language_probability"),
            )

            # Generate subtitle formats
            subtitle_formatter = SubtitleFormatter()
            srt_content = subtitle_formatter.to_srt(segments)
            vtt_content = subtitle_formatter.to_vtt(segments)
            txt_content = transcription.get("text", "")

            # Create ASR result if requested
            asr_result = None
            if request.asr_format:
                asr_result = self._create_asr_result(transcription, request.model)

            # Store results
            result_data = {
                "json_result": json_result.dict(),
                "srt_content": srt_content,
                "vtt_content": vtt_content,
                "txt_content": txt_content,
                "asr_result": asr_result.dict() if asr_result else None,
                "ray_result": ray_result,
            }

            await self.db.store_result(task_id, result_data)
            print(f"✅ Ray results stored successfully for task {task_id}")

        except Exception as e:
            print(f"❌ Error storing Ray results for task {task_id}: {e}")
            raise

    def _create_asr_result(self, transcription: dict, model: str) -> ASRModel:
        """Create ASR format result from Ray transcription."""
        try:
            events = []

            for word in transcription.get("words", []):
                event = EventModel(
                    content=word["text"],
                    start_time=word["start"],
                    end_time=word["end"],
                    event_type="word",
                    language=transcription.get("language"),
                    confidence=word.get("confidence"),
                    speaker=word.get("speaker"),
                    is_eol=False,
                    is_eos=False,
                )
                events.append(event)

            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service-ray",
                version=1,
                events=events,
                language=transcription.get("language"),
                language_probability=transcription.get("language_probability"),
                duration=transcription.get("duration"),
                processing_info={
                    "model": model,
                    "segments_count": len(transcription.get("segments", [])),
                    "words_count": len(transcription.get("words", [])),
                    "processing_mode": "ray_actors",
                },
            )

        except Exception as e:
            print(f"Error creating ASR result: {e}")
            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service-ray",
                version=1,
                events=[],
                language=transcription.get("language", "en"),
                language_probability=transcription.get("language_probability", 0.5),
            )

    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using Ray Actor."""
        temp_path = None

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

            print(f"🔍 Starting Ray language detection for file: {file_id}")

            # Ensure Ray is initialized
            if not ray.is_initialized():
                try:
                    ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
                except:
                    ray.init(ignore_reinit_error=True)

            # Use Ray Actor for language detection
            from src.services.ray_actors import WhisperTranscriptionActor

            # Create language detection actor (CPU only for speed)
            lang_actor = WhisperTranscriptionActor.remote(
                model_size="base", device="cpu"  # Use CPU for quick language detection
            )

            # Detect language
            result_ref = lang_actor.detect_language.remote(str(temp_path))
            result = ray.get(result_ref)

            return languageDetectionModel(
                file_id=file_id,
                language=result["language"],
                confidence=result["confidence"],
            )

        except Exception as e:
            print(f"❌ Ray language detection error: {e}")
            # Return fallback result
            return languageDetectionModel(
                file_id=file_id, language="en", confidence=0.5
            )
        finally:
            # Cleanup temp file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

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
        # For Ray tasks, we could implement cancellation by maintaining
        # a registry of running Ray tasks and calling ray.cancel()
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )
        return success

    # Result methods (same as before)
    async def get_json_result(self, task_id: str) -> Optional[JSONModel]:
        """Get JSON result for task."""
        try:
            result = await self.db.get_result(task_id)
            if not result or "json_result" not in result.get("result_data", {}):
                return None
            return JSONModel(**result["result_data"]["json_result"])
        except Exception as e:
            print(f"Error getting JSON result: {e}")
            return None

    async def get_asr_result(self, task_id: str) -> Optional[ASRModel]:
        """Get ASR result for task."""
        try:
            result = await self.db.get_result(task_id)
            if not result:
                return None

            result_data = result.get("result_data", {})
            asr_data = result_data.get("asr_result")

            if not asr_data:
                # Generate ASR on the fly if not available
                ray_result = result_data.get("ray_result", {})
                if ray_result and ray_result.get("transcription"):
                    asr_result = self._create_asr_result(
                        ray_result["transcription"], "base"
                    )
                    return asr_result
                return None

            return ASRModel(**asr_data)
        except Exception as e:
            print(f"Error getting ASR result: {e}")
            return None

    async def get_srt_result(self, task_id: str) -> Optional[str]:
        """Get SRT result."""
        try:
            result = await self.db.get_result(task_id)
            if not result:
                return None

            result_data = result.get("result_data", {})
            srt_content = result_data.get("srt_content")

            if not srt_content:
                # Generate SRT on the fly if not available
                json_result = result_data.get("json_result")
                if json_result:
                    from src.utils.subtitle_formats import SubtitleFormatter

                    json_model = JSONModel(**json_result)
                    subtitle_formatter = SubtitleFormatter()
                    srt_content = subtitle_formatter.to_srt(json_model.segments)
                    return srt_content
                return ""

            return srt_content
        except Exception as e:
            print(f"Error getting SRT result: {e}")
            return None

    async def get_vtt_result(self, task_id: str) -> Optional[str]:
        """Get VTT result."""
        try:
            result = await self.db.get_result(task_id)
            if not result:
                return None

            result_data = result.get("result_data", {})
            vtt_content = result_data.get("vtt_content")

            if not vtt_content:
                # Generate VTT on the fly if not available
                json_result = result_data.get("json_result")
                if json_result:
                    from src.utils.subtitle_formats import SubtitleFormatter

                    json_model = JSONModel(**json_result)
                    subtitle_formatter = SubtitleFormatter()
                    vtt_content = subtitle_formatter.to_vtt(json_model.segments)
                    return vtt_content
                return ""

            return vtt_content
        except Exception as e:
            print(f"Error getting VTT result: {e}")
            return None

    async def get_txt_result(self, task_id: str) -> Optional[str]:
        """Get TXT result."""
        try:
            result = await self.db.get_result(task_id)
            if not result:
                return None

            result_data = result.get("result_data", {})
            txt_content = result_data.get("txt_content")

            if not txt_content:
                # Generate TXT on the fly if not available
                json_result = result_data.get("json_result")
                if json_result:
                    return json_result.get("text", "")

                ray_result = result_data.get("ray_result")
                if ray_result and ray_result.get("transcription"):
                    return ray_result["transcription"].get("text", "")

                return ""

            return txt_content
        except Exception as e:
            print(f"Error getting TXT result: {e}")
            return None

    # URL transcription
    async def start_transcription_from_url(
        self, request: TranscriptionURLReqModel
    ) -> str:
        """Start transcription from URL."""
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(request.url)
                response.raise_for_status()

                # Get filename from URL or use default
                filename = Path(request.url).name or f"downloaded_{task_id[:8]}"
                if not filename.endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac")):
                    filename += ".mp3"  # Default extension

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

            # Process transcription using Ray
            await self._process_transcription_with_ray(task_id, transcription_request)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ URL transcription failed for task {task_id}: {error_msg}")
            await self.db.update_task(
                task_id, {"status": "failed", "error_message": error_msg}
            )


# For backward compatibility
TranscriptionService = RayTranscriptionService
