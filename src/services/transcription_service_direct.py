"""
src/services/transcription_service_direct.py

Direct transcription service that processes audio without Ray tasks.
This avoids dependency issues with Ray environments.
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
from src.models.pydantic_models import JSONModel, ASRModel, WordModel, SegmentModel, EventModel
from src.database.mongodb import MongoDB
from src.utils.ray_client import RayClient


class DirectTranscriptionService:
    """
    Direct transcription service that processes audio using subprocess calls
    instead of Ray tasks to avoid dependency issues.
    """

    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        """Start a new transcription task using direct processing."""
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
        asyncio.create_task(self._process_transcription_direct(task_id, request))

        return task_id

    async def _process_transcription_direct(
        self, task_id: str, request: TranscriptionReqModel
    ):
        """Process transcription using a direct Python subprocess."""
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

            # Create temp file for processing
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"{task_id}_{filename}"

            with open(temp_path, "wb") as f:
                f.write(file_data)

            print(f"🚀 Starting direct transcription: {task_id}")
            print(f"📁 Temp file: {temp_path} ({len(file_data)} bytes)")

            # Create a Python script for transcription
            script_content = self._create_transcription_script(
                str(temp_path), request
            )
            
            # Write script to temp file
            script_path = temp_dir / f"transcribe_{task_id}.py"
            with open(script_path, "w") as f:
                f.write(script_content)

            print(f"📝 Created transcription script: {script_path}")

            # Execute the transcription script
            cmd = [
                "python",
                str(script_path)
            ]

            print(f"🔧 Executing: {' '.join(cmd)}")

            # Run the transcription
            process = subprocess.run(
                cmd,
                cwd="/app",
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env={
                    **os.environ,
                    "PYTHONPATH": "/app:/app/src",
                }
            )

            if process.returncode == 0:
                # Parse the result from stdout
                result_lines = process.stdout.strip().split('\n')
                result_json = None
                
                for line in result_lines:
                    if line.startswith('RESULT_JSON:'):
                        result_json = json.loads(line[12:])
                        break

                if result_json:
                    print(f"✅ Transcription completed successfully: {task_id}")
                    
                    # Store results in database
                    await self._store_transcription_results(task_id, result_json, request)

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
                error_msg = f"Transcription script failed: {process.stderr}"
                print(f"❌ Script stderr: {process.stderr}")
                print(f"❌ Script stdout: {process.stdout}")
                raise Exception(error_msg)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Direct transcription failed for task {task_id}: {error_msg}")

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

    def _create_transcription_script(self, audio_path: str, request: TranscriptionReqModel) -> str:
        """Create a Python script for transcription."""
        
        return f'''
import sys
import os
import json

# Add paths
sys.path.append("/app")
sys.path.append("/app/src")

def main():
    try:
        print("🎯 Starting transcription script")
        print(f"   Audio: {audio_path}")
        print(f"   Model: {request.model}")
        print(f"   GPU: {request.gpu}")
        
        # Import libraries
        from faster_whisper import WhisperModel
        import torch
        
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
'''

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
                    is_eos=False
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
                    "words_count": len(result.get("words", []))
                }
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
                language_probability=result.get("language_probability", 0.5)
            )

    # Language detection and other methods (same as simple version)
    async def detect_language(self, file_id: str) -> languageDetectionModel:
        """Detect language using direct processing."""
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
            script_content = f'''
import sys
import os
import json

sys.path.append("/app")
sys.path.append("/app/src")

try:
    from faster_whisper import WhisperModel
    
    model = WhisperModel("base", device="cpu", compute_type="int8", download_root="/app/models/whis