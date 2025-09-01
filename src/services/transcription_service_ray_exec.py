"""
src/services/transcription_service_ray_exec.py

Transcription service that executes Python scripts in the Ray container
which has all the ML dependencies installed.
"""

import asyncio
import uuid
import subprocess
import tempfile
import json
import os
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


class RayExecTranscriptionService:
    """
    Transcription service that executes scripts in the Ray container
    where all ML dependencies are available.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task using Ray container execution."""
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
        asyncio.create_task(
            self._process_transcription_in_ray_container(task_id, request)
        )

        return task_id

    async def _process_transcription_in_ray_container(
        self, task_id: str, request: TranscriptionReqModel
    ):
        """Process transcription by executing script in Ray container."""
        temp_path = None
        script_path = None

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

            # Create temp file for processing (in shared volume)
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"{task_id}_{filename}"

            with open(temp_path, "wb") as f:
                f.write(file_data)

            print(f"🚀 Starting Ray container transcription: {task_id}")
            print(f"📁 Temp file: {temp_path} ({len(file_data)} bytes)")

            # Create a Python script for transcription
            script_content = self._create_transcription_script(str(temp_path), request)

            # Write script to shared temp directory
            script_path = temp_dir / f"transcribe_{task_id}.py"
            with open(script_path, "w") as f:
                f.write(script_content)

            print(f"📝 Created transcription script: {script_path}")

            # Execute the transcription script in Ray container using docker exec
            cmd = [
                "docker",
                "exec",
                "transcription_ray_head",
                "python",
                f"/app/temp/transcribe_{task_id}.py",
            ]

            print(f"🔧 Executing in Ray container: {' '.join(cmd)}")

            # Run the transcription
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if process.returncode == 0:
                # Parse the result from stdout
                result_lines = process.stdout.strip().split("\n")
                result_json = None

                # Print all output for debugging
                print("📊 Ray container output:")
                for line in result_lines:
                    print(f"   {line}")
                    if line.startswith("RESULT_JSON:"):
                        result_json = json.loads(line[12:])

                if result_json:
                    print(f"✅ Transcription completed successfully: {task_id}")

                    # Store results in database
                    await self._store_transcription_results(
                        task_id, result_json, request
                    )

                    # Update task status
                    await self.db.update_task(
                        task_id,
                        {
                            "status": "completed",
                            "completed_at": datetime.utcnow(),
                            "result_summary": {
                                "language": result_json.get("language"),
                                "duration": result_json.get("duration", 0),
                                "words_count": len(result_json.get("words", [])),
                                "segments_count": len(result_json.get("segments", [])),
                            },
                        },
                    )
                else:
                    raise Exception("No result found in transcription output")
            else:
                error_msg = f"Ray container execution failed: {process.stderr}"
                print(f"❌ Ray container stderr: {process.stderr}")
                print(f"❌ Ray container stdout: {process.stdout}")
                raise Exception(error_msg)

        except Exception as e:
            error_msg = str(e)
            print(
                f"❌ Ray container transcription failed for task {task_id}: {error_msg}"
            )

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

            if script_path and script_path.exists():
                try:
                    script_path.unlink()
                    print(f"🧹 Cleaned up temp script: {script_path}")
                except Exception as e:
                    print(f"⚠️ Failed to cleanup temp script: {e}")

    def _create_transcription_script(
        self, audio_path: str, request: TranscriptionReqModel
    ) -> str:
        """Create a Python script for transcription."""

        return f"""
import sys
import os
import json

# Add paths
sys.path.append("/app")
sys.path.append("/app/src")

def main():
    try:
        print("🎯 Starting transcription script in Ray container")
        print(f"   Audio: {audio_path}")
        print(f"   Model: {request.model}")
        print(f"   GPU: {request.gpu}")
        
        # Import libraries - should work in Ray container
        from faster_whisper import WhisperModel
        import torch
        import numpy as np
        print("✅ All libraries imported successfully")
        
        # Check file
        if not os.path.exists("{audio_path}"):
            raise Exception("Audio file not found")
        
        file_size = os.path.getsize("{audio_path}")
        print(f"   File size: {{file_size}} bytes")
        
        # Initialize model
        device = "cuda" if {request.gpu} and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"🤖 Loading Whisper model {request.model} on {{device}}")
        model = WhisperModel(
            "{request.model}",
            device=device,
            compute_type=compute_type,
            download_root="/app/models/whisper"
        )
        
        print("✅ Model loaded successfully")
        
        # Transcribe
        language = {json.dumps(request.language if request.language != "auto" else None)}
        initial_prompt = {json.dumps(request.prompt)}
        
        print("🎧 Starting transcription...")
        segments, info = model.transcribe(
            "{audio_path}",
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"🔍 Language detected: {{info.language}} (confidence: {{info.language_probability}})")
        
        # Process results
        result_segments = []
        words = []
        full_text = ""
        
        segment_count = 0
        for segment in segments:
            segment_count += 1
            segment_words = []
            segment_text = segment.text.strip()
            
            if segment_count <= 5:
                print(f"   Segment {{segment_count}}: '{{segment_text[:50]}}...'")
            
            if segment_text:
                full_text += segment_text + " "
                
                if hasattr(segment, "words") and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip() if hasattr(word, 'word') else str(word).strip()
                        word_dict = {{
                            "start": float(word.start),
                            "end": float(word.end),
                            "text": word_text,
                            "confidence": getattr(word, "probability", None)
                        }}
                        words.append(word_dict)
                        segment_words.append(word_dict)
                
                result_segments.append({{
                    "start": float(segment.start),
                    "end": float(segment.end), 
                    "text": segment_text,
                    "words": segment_words
                }})
        
        print(f"📊 Results: {{segment_count}} segments, {{len(result_segments)}} non-empty, {{len(words)}} words")
        
        # Try without VAD if no results
        if len(result_segments) == 0:
            print("⚠️ No segments found, retrying without VAD...")
            segments_retry, info_retry = model.transcribe(
                "{audio_path}",
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=True,
                vad_filter=False
            )
            
            for segment in segments_retry:
                segment_words = []
                segment_text = segment.text.strip()
                
                if segment_text:
                    full_text += segment_text + " "
                    
                    if hasattr(segment, "words") and segment.words:
                        for word in segment.words:
                            word_text = word.word.strip() if hasattr(word, 'word') else str(word).strip()
                            word_dict = {{
                                "start": float(word.start),
                                "end": float(word.end),
                                "text": word_text,
                                "confidence": getattr(word, "probability", None)
                            }}
                            words.append(word_dict)
                            segment_words.append(word_dict)
                    
                    result_segments.append({{
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment_text,
                        "words": segment_words
                    }})
            
            print(f"🔄 Retry results: {{len(result_segments)}} segments, {{len(words)}} words")
        
        # Create final result
        result = {{
            "segments": result_segments,
            "words": words,
            "text": full_text.strip(),
            "language": info.language,
            "language_probability": float(info.language_probability) if info.language_probability else 0.0,
            "duration": sum(s["end"] - s["start"] for s in result_segments) if result_segments else 0,
            "model": "{request.model}"
        }}
        
        print(f"✅ Transcription completed successfully")
        print(f"   Text: '{{full_text.strip()[:100]}}{{'...' if len(full_text) > 100 else ''}}'")
        
        # Output result as JSON (will be parsed by parent process)
        print("RESULT_JSON:" + json.dumps(result))
        
    except Exception as e:
        print(f"❌ Transcription script failed: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

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
                "ray_result": result,
            }

            await self.db.store_result(task_id, result_data)

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
                generated_by="advanced-transcription-service",
                version=1,
                events=events,
                language=result.get("language"),
                language_probability=result.get("language_probability"),
                duration=result.get("duration"),
                processing_info={
                    "model": model,
                    "segments_count": len(result.get("segments", [])),
                    "words_count": len(result.get("words", [])),
                },
            )

        except Exception as e:
            print(f"Error creating ASR result: {e}")
            return ASRModel(
                asr_model=model,
                created_at=datetime.utcnow().isoformat(),
                generated_by="advanced-transcription-service",
                version=1,
                events=[],
                language=result.get("language", "en"),
                language_probability=result.get("language_probability", 0.5),
            )

    # Language detection using Ray container
    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using Ray container execution."""
        temp_path = None
        script_path = None

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

            # Create language detection script
            script_content = f"""
import sys
import os
import json

sys.path.append("/app")
sys.path.append("/app/src")

try:
    from faster_whisper import WhisperModel
    
    model = WhisperModel("base", device="cpu", compute_type="int8", download_root="/app/models/whisper")
    
    segments, info = model.transcribe(
        "{temp_path}",
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Consume first segment to trigger language detection
    next(segments, None)
    
    result = {{
        "language": info.language,
        "confidence": float(info.language_probability) if info.language_probability else 0.5
    }}
    
    print("RESULT_JSON:" + json.dumps(result))
    
except Exception as e:
    print(f"Language detection failed: {{e}}")
    result = {{"language": "en", "confidence": 0.5}}
    print("RESULT_JSON:" + json.dumps(result))
"""

            script_path = temp_dir / f"detect_lang_{file_id}.py"
            with open(script_path, "w") as f:
                f.write(script_content)

            # Execute script in Ray container
            cmd = [
                "docker",
                "exec",
                "transcription_ray_head",
                "python",
                f"/app/temp/detect_lang_{file_id}.py",
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if process.returncode == 0:
                # Parse result
                result_lines = process.stdout.strip().split("\n")
                for line in result_lines:
                    if line.startswith("RESULT_JSON:"):
                        result_json = json.loads(line[12:])
                        return languageDetectionModel(
                            file_id=file_id,
                            language=result_json["language"],
                            confidence=result_json["confidence"],
                        )

            # Fallback
            return languageDetectionModel(
                file_id=file_id, language="en", confidence=0.5
            )

        except Exception as e:
            print(f"Language detection error: {e}")
            return languageDetectionModel(
                file_id=file_id, language="en", confidence=0.5
            )
        finally:
            # Cleanup temp files
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            if script_path and script_path.exists():
                try:
                    script_path.unlink()
                except:
                    pass

    # Task management methods (same implementation as previous versions)
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

    # Result methods (same as previous versions)
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

    # URL transcription (simplified)
    async def start_transcription_from_url(
        self, request: TranscriptionURLReqModel
    ) -> str:
        """Start transcription from URL (simplified)."""
        raise Exception("URL transcription not implemented in ray exec version")


# For backward compatibility
TranscriptionService = RayExecTranscriptionService
