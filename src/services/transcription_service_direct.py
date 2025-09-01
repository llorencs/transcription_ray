"""
src/services/transcription_service_direct.py

Pure direct transcription service that processes audio directly in-process.
No Docker, Ray, or subprocess dependencies.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

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


class DirectTranscriptionService:
    """
    Pure direct transcription service that processes audio by importing
    and using ML libraries directly in the same process.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task using direct in-process processing."""
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

        # Process transcription asynchronously in this same process
        asyncio.create_task(self._process_transcription_in_process(task_id, request))

        return task_id

    async def _process_transcription_in_process(
        self, task_id: str, request: TranscriptionReqModel
    ):
        """Process transcription completely in the current Python process."""
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

            print(f"🚀 Starting in-process transcription: {task_id}")
            print(f"📁 Temp file: {temp_path} ({len(file_data)} bytes)")

            # Process transcription directly in this Python process
            result = await self._transcribe_in_process(str(temp_path), request)

            if result:
                print(f"✅ Transcription completed successfully: {task_id}")

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
                            "text_preview": result.get("text", "")[:100],
                        },
                    },
                )
            else:
                raise Exception("Transcription returned empty result")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ In-process transcription failed for task {task_id}: {error_msg}")
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

    async def _transcribe_in_process(
        self, audio_path: str, request: TranscriptionReqModel
    ) -> dict:
        """Run transcription directly in this Python process using imported libraries."""

        # Run the transcription in a separate thread to avoid blocking the async event loop
        import asyncio
        import concurrent.futures

        def _run_whisper_transcription():
            try:
                import os

                print("🎯 Starting in-process Whisper transcription")
                print(f"   Audio: {audio_path}")
                print(f"   Model: {request.model}")
                print(f"   GPU: {request.gpu}")

                # Check if file exists and is readable
                if not os.path.exists(audio_path):
                    raise Exception(f"Audio file not found: {audio_path}")

                file_size = os.path.getsize(audio_path)
                print(f"   File size: {file_size} bytes")

                if file_size == 0:
                    raise Exception("Audio file is empty")

                # Import ML libraries
                try:
                    from faster_whisper import WhisperModel
                    import torch

                    print("✅ Successfully imported faster_whisper and torch")
                except ImportError as e:
                    print(f"❌ Failed to import ML libraries: {e}")
                    raise Exception(
                        f"ML dependencies not available in API container: {e}"
                    )

                # Initialize model
                device = "cuda" if request.gpu and torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"

                print(f"🤖 Loading Whisper model '{request.model}' on {device}")
                print(f"   Compute type: {compute_type}")

                # Ensure models directory exists
                models_dir = Path("/app/models/whisper")
                models_dir.mkdir(parents=True, exist_ok=True)

                try:
                    model = WhisperModel(
                        request.model,
                        device=device,
                        compute_type=compute_type,
                        download_root=str(models_dir.parent),
                    )
                    print("✅ Whisper model loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load Whisper model: {e}")
                    raise Exception(
                        f"Could not load Whisper model '{request.model}': {e}"
                    )

                # Prepare transcription parameters
                language = request.language if request.language != "auto" else None
                initial_prompt = request.prompt

                print("🎧 Starting Whisper transcription...")
                print(f"   Language: {language or 'auto-detect'}")
                print(f"   Initial prompt: {initial_prompt or 'none'}")

                try:
                    # Transcribe with error handling
                    segments, info = model.transcribe(
                        audio_path,
                        language=language,
                        initial_prompt=initial_prompt,
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )

                    print(f"✅ Whisper transcription completed")
                    print(
                        f"🔍 Detected language: {info.language} (confidence: {info.language_probability:.3f})"
                    )

                except Exception as e:
                    print(f"❌ Whisper transcription failed: {e}")
                    raise Exception(f"Transcription process failed: {e}")

                # Process results
                result_segments = []
                words = []
                full_text = ""

                print("📝 Processing transcription segments...")
                segment_count = 0

                try:
                    for segment in segments:
                        segment_count += 1
                        segment_words = []
                        segment_text = segment.text.strip()

                        # Show progress for first few segments
                        if segment_count <= 5:
                            print(
                                f"   Segment {segment_count}: '{segment_text[:50]}{'...' if len(segment_text) > 50 else ''}'"
                            )
                        elif segment_count == 6:
                            print("   ... (processing remaining segments)")

                        if segment_text:  # Only process non-empty segments
                            full_text += segment_text + " "

                            # Process words in segment
                            if hasattr(segment, "words") and segment.words:
                                for word in segment.words:
                                    try:
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
                                    except Exception as word_error:
                                        print(f"⚠️ Error processing word: {word_error}")
                                        continue

                            # Create segment
                            try:
                                result_segments.append(
                                    {
                                        "start": float(segment.start),
                                        "end": float(segment.end),
                                        "text": segment_text,
                                        "words": segment_words,
                                    }
                                )
                            except Exception as segment_error:
                                print(f"⚠️ Error processing segment: {segment_error}")
                                continue

                    print(f"📊 Processing completed:")
                    print(f"   Total segments processed: {segment_count}")
                    print(f"   Non-empty segments: {len(result_segments)}")
                    print(f"   Total words: {len(words)}")
                    print(f"   Full text length: {len(full_text)} characters")

                except Exception as e:
                    print(f"❌ Error processing segments: {e}")
                    # Continue with whatever we managed to process

                # Retry without VAD if no results
                if len(result_segments) == 0:
                    print(
                        "⚠️ No segments found with VAD, retrying without VAD filter..."
                    )
                    try:
                        segments_retry, info_retry = model.transcribe(
                            audio_path,
                            language=language,
                            initial_prompt=initial_prompt,
                            word_timestamps=True,
                            vad_filter=False,  # Disable VAD filter
                        )

                        print("📝 Processing retry segments...")
                        for segment in segments_retry:
                            segment_words = []
                            segment_text = segment.text.strip()

                            if segment_text:
                                full_text += segment_text + " "

                                if hasattr(segment, "words") and segment.words:
                                    for word in segment.words:
                                        try:
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
                                        except:
                                            continue

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

                    except Exception as retry_error:
                        print(f"❌ Retry without VAD also failed: {retry_error}")
                        # Return what we have, even if empty

                # Create final result
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
                    "model": request.model,
                }

                # Log final results
                final_text = result["text"]
                print(f"✅ Transcription processing completed successfully")
                print(
                    f"   Final text: '{final_text[:100]}{'...' if len(final_text) > 100 else ''}'"
                )
                print(f"   Language: {result['language']}")
                print(f"   Duration: {result['duration']:.2f}s")

                # Validate results
                if len(words) == 0 and len(result_segments) == 0:
                    print("⚠️ WARNING: No transcription results produced!")
                    print("   This could indicate:")
                    print("   - Audio file is silent or very quiet")
                    print("   - Audio format not supported")
                    print("   - Model failed to process the audio")

                    # Try basic audio analysis
                    try:
                        import librosa

                        y, sr = librosa.load(audio_path, sr=16000, duration=10.0)
                        duration = len(y) / sr
                        rms = librosa.feature.rms(y=y)[0].mean()
                        print(f"   Audio info: {duration:.2f}s, {sr}Hz, RMS: {rms:.6f}")

                        if duration < 0.5:
                            result["warning"] = "Audio file too short"
                        elif rms < 1e-6:
                            result["warning"] = "Audio appears to be silent"
                        else:
                            result["warning"] = "Unknown transcription issue"

                    except Exception as analysis_error:
                        print(f"   Audio analysis failed: {analysis_error}")
                        result["warning"] = "Could not analyze audio"

                return result

            except Exception as e:
                print(f"❌ Transcription thread failed: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                result = await loop.run_in_executor(
                    executor, _run_whisper_transcription
                )
                return result
            except Exception as e:
                print(f"❌ Transcription executor failed: {e}")
                raise

    async def _store_transcription_results(
        self, task_id: str, result: dict, request: TranscriptionReqModel
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
                for word in result.get("words", [])
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
                for segment in result.get("segments", [])
            ]

            json_result = JSONModel(
                text=result.get("text", ""),
                segments=segments,
                language=result.get("language"),
                language_probability=result.get("language_probability"),
            )

            # Generate subtitle formats
            subtitle_formatter = SubtitleFormatter()
            srt_content = subtitle_formatter.to_srt(segments)
            vtt_content = subtitle_formatter.to_vtt(segments)
            txt_content = result.get("text", "")

            # Create ASR result if requested
            asr_result = None
            if request.asr_format:
                asr_result = self._create_asr_result(result, request.model)

            # Store results
            result_data = {
                "json_result": json_result.dict(),
                "srt_content": srt_content,
                "vtt_content": vtt_content,
                "txt_content": txt_content,
                "asr_result": asr_result.dict() if asr_result else None,
                "ray_result": result,  # For compatibility
            }

            await self.db.store_result(task_id, result_data)
            print(f"✅ Results stored successfully for task {task_id}")

        except Exception as e:
            print(f"❌ Error storing results for task {task_id}: {e}")
            raise

    def _create_asr_result(self, result: dict, model: str) -> ASRModel:
        """Create ASR format result."""
        try:
            events = []

            for word in result.get("words", []):
                event = EventModel(
                    content=word["text"],
                    start_time=word["start"],
                    end_time=word["end"],
                    event_type="word",
                    language=result.get("language"),
                    confidence=word.get("confidence"),
                    speaker=word.get("speaker"),
                    is_eol=False,
                    is_eos=False,
                )
                events.append(event)

            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service-direct",
                version=1,
                events=events,
                language=result.get("language"),
                language_probability=result.get("language_probability"),
                duration=result.get("duration"),
                processing_info={
                    "model": model,
                    "segments_count": len(result.get("segments", [])),
                    "words_count": len(result.get("words", [])),
                    "processing_mode": "direct_in_process",
                },
            )

        except Exception as e:
            print(f"Error creating ASR result: {e}")
            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service-direct",
                version=1,
                events=[],
                language=result.get("language", "en"),
                language_probability=result.get("language_probability", 0.5),
            )

    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using direct in-process processing."""
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

            print(f"🔍 Starting language detection for file: {file_id}")

            # Run language detection in thread to avoid blocking
            import asyncio
            import concurrent.futures

            def _detect_language():
                try:
                    from faster_whisper import WhisperModel

                    print("✅ Imported faster_whisper for language detection")

                    # Use CPU for language detection to save GPU resources
                    models_dir = Path("/app/models/whisper")
                    models_dir.mkdir(parents=True, exist_ok=True)

                    model = WhisperModel(
                        "base",
                        device="cpu",
                        compute_type="int8",
                        download_root=str(models_dir.parent),
                    )

                    print("✅ Language detection model loaded on CPU")

                    # Quick language detection - just need to trigger detection
                    segments, info = model.transcribe(
                        str(temp_path),
                        language=None,  # Auto-detect
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )

                    # Consume first segment to trigger language detection
                    try:
                        next(segments, None)
                    except:
                        pass  # Ignore if no segments

                    return {
                        "language": info.language,
                        "confidence": info.language_probability or 0.5,
                    }

                except ImportError as e:
                    print(f"❌ Failed to import faster_whisper: {e}")
                    return {"language": "en", "confidence": 0.5}
                except Exception as e:
                    print(f"❌ Language detection failed: {e}")
                    return {"language": "en", "confidence": 0.5}

            # Run in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                lang_result = await loop.run_in_executor(executor, _detect_language)

            result = languageDetectionModel(
                file_id=file_id,
                language=lang_result["language"],
                confidence=lang_result["confidence"],
            )

            print(
                f"✅ Language detected: {result.language} (confidence: {result.confidence})"
            )
            return result

        except Exception as e:
            print(f"❌ Language detection error: {e}")
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
        success = await self.db.update_task(
            task_id, {"status": "cancelled", "cancelled_at": datetime.utcnow()}
        )
        return success

    # Result methods
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
                if ray_result:
                    asr_result = self._create_asr_result(ray_result, "base")
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
                if ray_result:
                    return ray_result.get("text", "")

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

            # Process transcription directly
            await self._process_transcription_in_process(task_id, transcription_request)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ URL transcription failed for task {task_id}: {error_msg}")
            await self.db.update_task(
                task_id, {"status": "failed", "error_message": error_msg}
            )


# For backward compatibility
TranscriptionService = DirectTranscriptionService
