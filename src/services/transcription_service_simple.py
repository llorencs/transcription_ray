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
import traceback

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


class SimpleTranscriptionService:
    """
    Simplified transcription service using Ray tasks instead of complex Jobs.
    More reliable and easier to debug.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client
        self._ensure_ray_initialized()

    def _ensure_ray_initialized(self):
        """Ensure Ray is initialized."""
        if not ray.is_initialized():
            try:
                ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
                print("✅ Ray initialized in SimpleTranscriptionService")
            except Exception as e:
                print(f"⚠️ Failed to connect to Ray cluster: {e}")
                print("⚠️ Initializing Ray locally as fallback")
                try:
                    ray.init(ignore_reinit_error=True)
                    print("✅ Ray initialized locally")
                except Exception as local_e:
                    print(f"❌ Failed to initialize Ray locally: {local_e}")
                    raise

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
            temp_dir.mkdir(exist_ok=True, parents=True)
            temp_path = temp_dir / f"{task_id}_{filename}"

            print(f"📝 Writing temp file: {temp_path}")
            with open(temp_path, "wb") as f:
                f.write(file_data)

            # Verify the file was written correctly
            if not temp_path.exists():
                raise Exception("Failed to create temporary file")

            file_size = temp_path.stat().st_size
            print(f"📁 Temp file created: {temp_path} ({file_size} bytes)")

            if file_size == 0:
                raise Exception("Temporary file is empty")

            print(f"🚀 Starting Ray task for transcription: {task_id}")

            # Ensure Ray is initialized before creating the remote function
            self._ensure_ray_initialized()

            # Create Ray task for transcription
            @ray.remote(
                num_gpus=1 if request.gpu else 0,
                num_cpus=2,
                memory=4_000_000_000,
                max_retries=1,
            )
            def transcription_task(audio_path: str, config: dict):
                try:
                    import sys
                    import os

                    sys.path.insert(0, "/app/src")
                    sys.path.insert(0, "/app")

                    print(f"🎯 Processing audio: {audio_path}")
                    print(f"🤖 Model: {config['model']}")
                    print(f"🔧 GPU: {config['use_gpu']}")

                    # Verify audio file exists and is readable
                    if not os.path.exists(audio_path):
                        raise Exception(f"Audio file not found: {audio_path}")

                    file_size = os.path.getsize(audio_path)
                    print(f"📁 Audio file size: {file_size} bytes")

                    if file_size == 0:
                        raise Exception("Audio file is empty")

                    # Import models after verifying file
                    from faster_whisper import WhisperModel
                    import torch
                    import librosa
                    import numpy as np  # FIXED: Added numpy import here

                    # Test audio file loading first
                    try:
                        print(f"🔍 Testing audio file loading...")
                        y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
                        print(f"   Audio loaded: {len(y)} samples, {sr}Hz")

                        if len(y) == 0:
                            raise Exception("Audio file contains no data")

                        duration_test = len(y) / sr
                        print(f"   Test duration: {duration_test:.2f}s")

                        # Calculate RMS using numpy
                        rms = np.sqrt(np.mean(y**2))
                        print(f"   RMS: {rms:.6f}")

                        if rms < 1e-6:
                            print(
                                "⚠️  WARNING: Audio appears to be very quiet or silent"
                            )

                    except Exception as audio_test_error:
                        print(f"❌ Audio loading test failed: {audio_test_error}")
                        raise Exception(
                            f"Audio file cannot be processed: {audio_test_error}"
                        )

                    # Initialize Whisper
                    device = (
                        "cuda"
                        if config["use_gpu"] and torch.cuda.is_available()
                        else "cpu"
                    )
                    compute_type = "float16" if device == "cuda" else "int8"

                    print(f"🤖 Loading Whisper model {config['model']} on {device}...")
                    model = WhisperModel(
                        config["model"],
                        device=device,
                        compute_type=compute_type,
                        download_root="/app/models/whisper",
                    )

                    print(f"✅ Whisper model loaded on {device}")

                    # Transcribe with word timestamps
                    print(f"🎧 Starting Whisper transcription...")
                    segments, info = model.transcribe(
                        audio_path,
                        language=config.get("language"),
                        initial_prompt=config.get("initial_prompt"),
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )

                    print(
                        f"🔍 Language detected: {info.language} (confidence: {info.language_probability})"
                    )

                    # Convert to our format
                    result_segments = []
                    words = []
                    full_text = ""
                    segment_count = 0

                    print(f"📝 Processing transcription segments...")
                    for segment in segments:
                        segment_count += 1
                        segment_words = []
                        segment_text = segment.text.strip()

                        if segment_count <= 5:  # Only print first 5 segments
                            print(
                                f"   Segment {segment_count}: '{segment_text[:50]}...' ({segment.start:.2f}-{segment.end:.2f}s)"
                            )

                        if segment_text:  # Only add non-empty segments
                            full_text += segment_text + " "

                            if hasattr(segment, "words") and segment.words:
                                for word in segment.words:
                                    word_text = (
                                        word.word.strip()
                                        if hasattr(word, "word")
                                        else str(word).strip()
                                    )
                                    word_dict = {
                                        "start": float(word.start),
                                        "end": float(word.end),
                                        "text": word_text,
                                        "confidence": getattr(
                                            word, "probability", None
                                        ),
                                    }
                                    words.append(word_dict)
                                    segment_words.append(word_dict)

                            result_segments.append(
                                {
                                    "start": float(segment.start),
                                    "end": float(segment.end),
                                    "text": segment_text,
                                    "words": segment_words,
                                }
                            )

                    print(f"📊 Transcription summary:")
                    print(f"   - Total segments processed: {segment_count}")
                    print(f"   - Non-empty segments: {len(result_segments)}")
                    print(f"   - Total words: {len(words)}")
                    print(f"   - Full text length: {len(full_text)} chars")

                    if len(result_segments) == 0:
                        print("⚠️  WARNING: No segments with content found!")
                        print("🔄 Retrying without VAD filter...")

                        # Try without VAD filter as fallback
                        segments_retry, info_retry = model.transcribe(
                            audio_path,
                            language=config.get("language"),
                            initial_prompt=config.get("initial_prompt"),
                            word_timestamps=True,
                            vad_filter=False,
                        )

                        for segment in segments_retry:
                            segment_count += 1
                            segment_words = []
                            segment_text = segment.text.strip()

                            if segment_text:
                                full_text += segment_text + " "

                                if hasattr(segment, "words") and segment.words:
                                    for word in segment.words:
                                        word_text = (
                                            word.word.strip()
                                            if hasattr(word, "word")
                                            else str(word).strip()
                                        )
                                        word_dict = {
                                            "start": float(word.start),
                                            "end": float(word.end),
                                            "text": word_text,
                                            "confidence": getattr(
                                                word, "probability", None
                                            ),
                                        }
                                        words.append(word_dict)
                                        segment_words.append(word_dict)

                                result_segments.append(
                                    {
                                        "start": float(segment.start),
                                        "end": float(segment.end),
                                        "text": segment_text,
                                        "words": segment_words,
                                    }
                                )

                        print(
                            f"🔄 Retry results: {len(result_segments)} segments, {len(words)} words"
                        )

                    result = {
                        "segments": result_segments,
                        "words": words,
                        "text": full_text.strip(),
                        "language": info.language,
                        "language_probability": (
                            float(info.language_probability)
                            if info.language_probability
                            else 0.0
                        ),
                        "duration": (
                            sum(s["end"] - s["start"] for s in result_segments)
                            if result_segments
                            else 0
                        ),
                        "model": config["model"],
                    }

                    print(f"✅ Transcription completed successfully")
                    print(f"   - Final segments: {len(result_segments)}")
                    print(f"   - Final words: {len(words)}")

                    # Add some debug info if no content was transcribed
                    if len(words) == 0:
                        print(
                            "❌ ERROR: No words transcribed! This indicates a serious problem."
                        )
                        print("   Possible causes:")
                        print("   - Audio file is corrupted or unreadable")
                        print("   - Audio has no speech content")
                        print("   - Whisper model failed to load properly")
                        print("   - File format incompatible")

                        # Add some basic audio info
                        try:
                            y_full, sr_full = librosa.load(audio_path, sr=None)
                            duration = len(y_full) / sr_full
                            rms_full = np.sqrt(np.mean(y_full**2))
                            print(
                                f"   Audio file info: {duration:.2f}s, {sr_full}Hz, {len(y_full)} samples"
                            )
                            print(f"   RMS energy: {rms_full:.6f}")
                        except Exception as audio_error:
                            print(f"   Could not analyze audio: {audio_error}")

                    return result

                except Exception as e:
                    print(f"❌ Ray task failed: {e}")
                    import traceback

                    traceback.print_exc()
                    raise

            # Submit Ray task
            task_config = {
                "model": request.model,
                "language": request.language if request.language != "auto" else None,
                "initial_prompt": request.prompt,
                "use_gpu": request.gpu,
            }

            future = transcription_task.remote(str(temp_path), task_config)

            # Get result with timeout
            try:
                result = ray.get(future, timeout=600)  # 10 minute timeout
            except ray.exceptions.RayTaskError as e:
                print(f"❌ Ray task error: {e}")
                raise Exception(f"Transcription task failed: {str(e)}")
            except ray.exceptions.GetTimeoutError:
                print(f"❌ Ray task timeout after 10 minutes")
                raise Exception("Transcription task timed out")

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

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Transcription processing failed for task {task_id}: {error_msg}")
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
            # Always cleanup temp file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    print(f"🧹 Cleaned up temp file: {temp_path}")
                except Exception as e:
                    print(f"⚠️ Failed to cleanup temp file: {e}")

    async def _store_transcription_results(
        self, task_id: str, ray_result: dict, request: TranscriptionReqModel
    ):
        """Store transcription results in database."""
        from src.utils.subtitle_formats import SubtitleFormatter

        try:
            # Convert to our models with proper error handling
            words = []
            for word in ray_result.get("words", []):
                try:
                    words.append(
                        WordModel(
                            start=float(word["start"]),
                            end=float(word["end"]),
                            text=str(word["text"]),
                            confidence=(
                                float(word["confidence"])
                                if word.get("confidence") is not None
                                else None
                            ),
                            speaker=(
                                str(word["speaker"]) if word.get("speaker") else None
                            ),
                        )
                    )
                except Exception as e:
                    print(f"⚠️ Error processing word: {e}, word data: {word}")
                    continue

            segments = []
            for segment in ray_result.get("segments", []):
                try:
                    segment_words = []
                    for word in segment.get("words", []):
                        try:
                            segment_words.append(
                                WordModel(
                                    start=float(word["start"]),
                                    end=float(word["end"]),
                                    text=str(word["text"]),
                                    confidence=(
                                        float(word["confidence"])
                                        if word.get("confidence") is not None
                                        else None
                                    ),
                                    speaker=(
                                        str(word["speaker"])
                                        if word.get("speaker")
                                        else None
                                    ),
                                )
                            )
                        except Exception as e:
                            print(f"⚠️ Error processing segment word: {e}")
                            continue

                    segments.append(
                        SegmentModel(
                            start=float(segment["start"]),
                            end=float(segment["end"]),
                            text=str(segment["text"]),
                            words=segment_words,
                        )
                    )
                except Exception as e:
                    print(f"⚠️ Error processing segment: {e}, segment data: {segment}")
                    continue

            json_result = JSONModel(
                text=str(ray_result.get("text", "")),
                segments=segments,
                language=str(ray_result.get("language", "en")),
                language_probability=float(ray_result.get("language_probability", 0.0)),
            )

            # Generate subtitle formats
            subtitle_formatter = SubtitleFormatter()
            srt_content = subtitle_formatter.to_srt(segments) if segments else ""
            vtt_content = subtitle_formatter.to_vtt(segments) if segments else ""
            txt_content = str(ray_result.get("text", ""))

            # Create ASR result if requested
            asr_result = None
            if request.asr_format:
                asr_result = self._create_asr_result(ray_result, request.model)

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
            print(f"✅ Results stored for task {task_id}")

        except Exception as e:
            print(f"❌ Error storing results for task {task_id}: {e}")
            traceback.print_exc()
            raise

    def _create_asr_result(self, ray_result: dict, model: str) -> ASRModel:
        """Create ASR format result from ray result."""
        try:
            # Convert words to events
            events = []

            for word in ray_result.get("words", []):
                try:
                    event = EventModel(
                        content=str(word["text"]),
                        start_time=float(word["start"]),
                        end_time=float(word["end"]),
                        event_type="word",
                        language=str(ray_result.get("language", "en")),
                        confidence=(
                            float(word["confidence"])
                            if word.get("confidence") is not None
                            else None
                        ),
                        speaker=str(word["speaker"]) if word.get("speaker") else None,
                        is_eol=False,
                        is_eos=False,
                    )
                    events.append(event)
                except Exception as e:
                    print(f"⚠️ Error creating event from word: {e}")
                    continue

            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service",
                version=1,
                events=events,
                language=str(ray_result.get("language", "en")),
                language_probability=float(ray_result.get("language_probability", 0.0)),
                duration=float(ray_result.get("duration", 0.0)),
                processing_info={
                    "model": model,
                    "segments_count": len(ray_result.get("segments", [])),
                    "words_count": len(ray_result.get("words", [])),
                },
            )

        except Exception as e:
            print(f"Error creating ASR result: {e}")
            traceback.print_exc()
            # Return empty ASR result as fallback
            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service",
                version=1,
                events=[],
                language="en",
                language_probability=0.0,
            )

    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using simple Ray task."""
        temp_path = None
        try:
            # Get file
            file_data, filename = await self.db.get_file(file_id)
            if not file_data:
                raise Exception("File not found")

            # Create temp file
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True, parents=True)
            temp_path = temp_dir / f"lang_{file_id}_{filename}"

            with open(temp_path, "wb") as f:
                f.write(file_data)

            # Ensure Ray is initialized
            self._ensure_ray_initialized()

            @ray.remote(num_cpus=1, memory=2_000_000_000, max_retries=1)
            def language_detection_task(audio_path: str):
                try:
                    import sys

                    sys.path.insert(0, "/app/src")
                    sys.path.insert(0, "/app")

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
                        "confidence": (
                            float(info.language_probability)
                            if info.language_probability
                            else 0.5
                        ),
                    }

                except Exception as e:
                    print(f"Language detection failed: {e}")
                    return {"language": "en", "confidence": 0.5}

            # Submit and get result
            future = language_detection_task.remote(str(temp_path))
            result = ray.get(future, timeout=60)  # 1 minute timeout

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
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )
        return success

    # Result retrieval methods
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
                # Generate ASR result on the fly if not available
                ray_result = result_data.get("ray_result", {})
                if ray_result:
                    asr_result = self._create_asr_result(ray_result, "base")
                    return asr_result
                return None

            return ASRModel(**asr_data)
        except Exception as e:
            print(f"Error getting ASR result: {e}")
            traceback.print_exc()
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
            traceback.print_exc()
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
            traceback.print_exc()
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
                # Try to get from json_result or ray_result
                json_result = result_data.get("json_result")
                if json_result:
                    txt_content = json_result.get("text", "")
                else:
                    ray_result = result_data.get("ray_result")
                    if ray_result:
                        txt_content = ray_result.get("text", "")

            return txt_content or ""

        except Exception as e:
            print(f"Error getting TXT result: {e}")
            traceback.print_exc()
            return None

    async def start_transcription_from_url(
        self, request: TranscriptionURLReqModel
    ) -> str:
        """Start transcription from URL (simplified)."""
        # For now, return error - implement if needed
        raise Exception("URL transcription not implemented in simple version")


# For backward compatibility
TranscriptionService = SimpleTranscriptionService
