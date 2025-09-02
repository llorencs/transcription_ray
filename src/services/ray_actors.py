"""
Ray actors for distributed transcription processing.
Updated to handle import issues and provide better error messages.
"""

import os
import ray
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add current path to ensure imports work
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")

from src.models.pydantic_models import (
    WordModel,
    SegmentModel,
    JSONModel,
    ASRModel,
    EventModel,
)


@ray.remote
class WhisperTranscriptionActor:
    """Ray actor for Whisper transcription with better error handling."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self._initialized = False
        self._load_model()

    def _load_model(self):
        """Load the Whisper model with comprehensive error handling."""
        try:
            # Test imports first
            print(f"[WhisperActor] Testing imports...")

            try:
                import torch

                print(f"[WhisperActor] ✅ torch imported: {torch.__version__}")
            except ImportError as e:
                print(f"[WhisperActor] ❌ torch import failed: {e}")
                raise ImportError(f"torch not available: {e}")

            try:
                import librosa

                print(f"[WhisperActor] ✅ librosa imported: {librosa.__version__}")
            except ImportError as e:
                print(f"[WhisperActor] ❌ librosa import failed: {e}")
                raise ImportError(f"librosa not available: {e}")

            try:
                import soundfile as sf

                print(f"[WhisperActor] ✅ soundfile imported")
            except ImportError as e:
                print(f"[WhisperActor] ❌ soundfile import failed: {e}")
                raise ImportError(f"soundfile not available: {e}")

            try:
                from faster_whisper import WhisperModel

                print(f"[WhisperActor] ✅ faster_whisper imported")
            except ImportError as e:
                print(f"[WhisperActor] ❌ faster_whisper import failed: {e}")
                raise ImportError(f"faster_whisper not available: {e}")

            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    print(f"[WhisperActor] Using CUDA device")
                else:
                    device = "cpu"
                    compute_type = "int8"
                    print(f"[WhisperActor] Using CPU device")
            else:
                device = self.device
                compute_type = "float16" if device == "cuda" else "int8"

            # Create models directory
            models_dir = Path("/app/models/whisper")
            models_dir.mkdir(parents=True, exist_ok=True)

            print(f"[WhisperActor] Loading Whisper model {self.model_size} on {device}")

            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root="/app/models/whisper",
            )

            print(
                f"[WhisperActor] ✅ Loaded Whisper model {self.model_size} on {device}"
            )
            self._initialized = True

        except Exception as e:
            print(f"[WhisperActor] ❌ Failed to load Whisper model: {e}")
            import traceback

            traceback.print_exc()
            self._initialized = False
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict], str]:
        """Transcribe audio file with detailed error handling."""

        if not self._initialized:
            raise RuntimeError("WhisperActor not properly initialized")

        try:
            print(f"[WhisperActor] Starting transcription of: {audio_path}")

            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            file_size = os.path.getsize(audio_path)
            print(f"[WhisperActor] File size: {file_size} bytes")

            if file_size == 0:
                raise ValueError("Audio file is empty")

            # Transcribe with word timestamps
            print(f"[WhisperActor] Running Whisper transcription...")
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            print(f"[WhisperActor] Transcription completed")
            print(
                f"[WhisperActor] Detected language: {info.language} (confidence: {info.language_probability})"
            )

            # Convert to our format
            word_list = []
            segment_list = []
            full_text = ""

            segment_count = 0
            for segment in segments:
                segment_count += 1
                segment_words = []
                segment_text = segment.text.strip()

                if segment_text:
                    full_text += segment_text + " "

                    if hasattr(segment, "words") and segment.words:
                        for word in segment.words:
                            word_dict = {
                                "start": float(word.start),
                                "end": float(word.end),
                                "text": word.word.strip(),
                                "confidence": getattr(word, "probability", None),
                            }
                            word_list.append(word_dict)
                            segment_words.append(word_dict)

                    segment_dict = {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment_text,
                        "words": segment_words,
                    }
                    segment_list.append(segment_dict)

            print(
                f"[WhisperActor] Processed {segment_count} segments, {len(segment_list)} non-empty, {len(word_list)} words"
            )

            # If no results, try without VAD
            if len(segment_list) == 0:
                print(f"[WhisperActor] No segments found, retrying without VAD...")
                segments_retry, info_retry = self.model.transcribe(
                    audio_path,
                    language=language,
                    initial_prompt=initial_prompt,
                    word_timestamps=True,
                    vad_filter=False,
                )

                for segment in segments_retry:
                    segment_words = []
                    segment_text = segment.text.strip()

                    if segment_text:
                        full_text += segment_text + " "

                        if hasattr(segment, "words") and segment.words:
                            for word in segment.words:
                                word_dict = {
                                    "start": float(word.start),
                                    "end": float(word.end),
                                    "text": word.word.strip(),
                                    "confidence": getattr(word, "probability", None),
                                }
                                word_list.append(word_dict)
                                segment_words.append(word_dict)

                        segment_dict = {
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "text": segment_text,
                            "words": segment_words,
                        }
                        segment_list.append(segment_dict)

                print(
                    f"[WhisperActor] Retry without VAD: {len(segment_list)} segments, {len(word_list)} words"
                )

            result = {
                "words": word_list,
                "segments": segment_list,
                "language": info.language,
                "language_probability": (
                    float(info.language_probability)
                    if info.language_probability
                    else 0.0
                ),
            }

            return result, full_text.strip()

        except Exception as e:
            print(f"[WhisperActor] ❌ Transcription failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def detect_language(self, audio_path: str) -> Dict[str, Any]:
        """Detect language of audio file."""

        if not self._initialized:
            raise RuntimeError("WhisperActor not properly initialized")

        try:
            print(f"[WhisperActor] Starting language detection: {audio_path}")

            # Check file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Load first 30 seconds for language detection
            segments, info = self.model.transcribe(
                audio_path,
                language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Consume first segment to get language info
            try:
                first_segment = next(segments, None)
            except:
                pass

            result = {
                "language": info.language,
                "confidence": (
                    float(info.language_probability)
                    if info.language_probability
                    else 0.5
                ),
            }

            print(f"[WhisperActor] Language detection result: {result}")
            return result

        except Exception as e:
            print(f"[WhisperActor] ❌ Language detection failed: {e}")
            import traceback

            traceback.print_exc()
            # Return fallback
            return {"language": "en", "confidence": 0.5}

    def get_status(self):
        """Get actor status for debugging."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "initialized": self._initialized,
            "model_loaded": self.model is not None,
        }


@ray.remote
class DiarizationActor:
    """Ray actor for speaker diarization."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self.pipeline = None
        self._initialized = False
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the diarization pipeline."""
        try:
            print(f"[DiarizationActor] Testing pyannote.audio import...")

            try:
                import torch

                print(f"[DiarizationActor] ✅ torch imported: {torch.__version__}")
            except ImportError as e:
                print(f"[DiarizationActor] ❌ torch import failed: {e}")
                raise ImportError(f"torch not available: {e}")

            try:
                from pyannote.audio import Pipeline
                from pyannote.audio.pipelines.utils.hook import ProgressHook

                print(f"[DiarizationActor] ✅ pyannote.audio imported")
            except ImportError as e:
                print(f"[DiarizationActor] ❌ pyannote.audio import failed: {e}")
                raise ImportError(f"pyannote.audio not available: {e}")

            # Use Hugging Face token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

            if not hf_token:
                print(
                    f"[DiarizationActor] ⚠️ No HUGGINGFACE_TOKEN found, diarization may not work"
                )

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
            )

            # Set device
            if self.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device)

            self.pipeline.to(device)

            print(f"[DiarizationActor] ✅ Loaded diarization pipeline on {device}")
            self._initialized = True

        except Exception as e:
            print(f"[DiarizationActor] ❌ Failed to load diarization pipeline: {e}")
            import traceback

            traceback.print_exc()
            self._initialized = False
            raise

    def diarize(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> Dict[str, List]:
        """Perform speaker diarization."""

        if not self._initialized:
            raise RuntimeError("DiarizationActor not properly initialized")

        try:
            print(f"[DiarizationActor] Starting diarization: {audio_path}")

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Run diarization
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            with ProgressHook() as hook:
                diarization = self.pipeline(
                    audio_path, hook=hook, num_speakers=num_speakers
                )

            # Convert to our format
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append(
                    {
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": speaker,
                    }
                )

            print(f"[DiarizationActor] Found {len(speaker_segments)} speaker segments")
            return {"speaker_segments": speaker_segments}

        except Exception as e:
            print(f"[DiarizationActor] ❌ Diarization failed: {e}")
            import traceback

            traceback.print_exc()
            raise


@ray.remote
class TranscriptionCoordinator:
    """Coordinator for orchestrating the full transcription pipeline."""

    def __init__(self):
        self.whisper_actor = None
        self.diarization_actor = None
        self._initialized = False

    def initialize_actors(self, model_size: str, use_gpu: bool = True):
        """Initialize required actors."""
        try:
            print(f"[TranscriptionCoordinator] Initializing actors...")
            print(f"[TranscriptionCoordinator] Model: {model_size}, GPU: {use_gpu}")

            device = "cuda" if use_gpu else "cpu"

            # Test actor creation
            print(f"[TranscriptionCoordinator] Creating WhisperTranscriptionActor...")
            self.whisper_actor = WhisperTranscriptionActor.remote(model_size, device)

            # Test the actor by getting its status
            try:
                status = ray.get(self.whisper_actor.get_status.remote(), timeout=30)
                print(f"[TranscriptionCoordinator] Whisper actor status: {status}")
                if not status.get("initialized"):
                    raise RuntimeError("Whisper actor failed to initialize")
            except Exception as e:
                print(f"[TranscriptionCoordinator] ❌ Whisper actor test failed: {e}")
                raise

            # Only create diarization actor if requested
            # self.diarization_actor = DiarizationActor.remote(device)

            print(f"[TranscriptionCoordinator] ✅ Actors initialized successfully")
            self._initialized = True

        except Exception as e:
            print(f"[TranscriptionCoordinator] ❌ Actor initialization failed: {e}")
            import traceback

            traceback.print_exc()
            self._initialized = False
            raise

    async def process_transcription(
        self,
        audio_path: str,
        model_size: str = "base",
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        diarize: bool = False,
        use_gpu: bool = True,
        max_segment_duration: float = 600.0,
    ) -> Dict[str, Any]:
        """Process complete transcription pipeline."""
        try:
            print(f"[TranscriptionCoordinator] Starting transcription pipeline")
            print(f"[TranscriptionCoordinator] Audio: {audio_path}")
            print(f"[TranscriptionCoordinator] Model: {model_size}, GPU: {use_gpu}")

            # Initialize actors if not done
            if not self._initialized or not self.whisper_actor:
                print(f"[TranscriptionCoordinator] Initializing actors...")
                self.initialize_actors(model_size, use_gpu)

            # Check if audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Get audio duration
            try:
                import librosa

                audio_duration = librosa.get_duration(filename=audio_path)
                print(
                    f"[TranscriptionCoordinator] Audio duration: {audio_duration:.2f}s"
                )
            except Exception as e:
                print(f"[TranscriptionCoordinator] Could not get audio duration: {e}")
                audio_duration = 0

            results = {
                "transcription": None,
                "diarization": None,
                "segments_processed": [audio_path],
                "processing_info": {
                    "model_size": model_size,
                    "language": language,
                    "diarization_enabled": diarize,
                    "gpu_used": use_gpu,
                    "started_at": datetime.utcnow().isoformat(),
                    "total_duration": audio_duration,
                    "segments_count": 1,
                },
            }

            # Process transcription
            print(f"[TranscriptionCoordinator] Running Whisper transcription...")

            try:
                transcription_result, segment_text = (
                    await self.whisper_actor.transcribe.remote(
                        audio_path, language, initial_prompt
                    )
                )
                print(f"[TranscriptionCoordinator] ✅ Transcription completed")
                print(
                    f"[TranscriptionCoordinator] Text length: {len(segment_text)} chars"
                )
                print(
                    f"[TranscriptionCoordinator] Words: {len(transcription_result.get('words', []))}"
                )
                print(
                    f"[TranscriptionCoordinator] Segments: {len(transcription_result.get('segments', []))}"
                )

            except Exception as e:
                print(
                    f"[TranscriptionCoordinator] ❌ Whisper transcription failed: {e}"
                )
                import traceback

                traceback.print_exc()
                raise

            # Store transcription results
            results["transcription"] = {
                "words": transcription_result["words"],
                "segments": transcription_result["segments"],
                "text": segment_text,
                "language": transcription_result.get("language"),
                "language_probability": transcription_result.get(
                    "language_probability"
                ),
            }

            # Perform diarization if requested (disabled for now to simplify)
            if diarize and False:  # Disabled for debugging
                print("[TranscriptionCoordinator] Performing speaker diarization...")
                if not self.diarization_actor:
                    device = "cuda" if use_gpu else "cpu"
                    self.diarization_actor = DiarizationActor.remote(device)

                try:
                    diarization_result = await self.diarization_actor.diarize.remote(
                        audio_path
                    )
                    results["diarization"] = diarization_result
                    print(f"[TranscriptionCoordinator] ✅ Diarization completed")
                except Exception as e:
                    print(f"[TranscriptionCoordinator] ⚠️ Diarization failed: {e}")

            results["processing_info"]["completed_at"] = datetime.utcnow().isoformat()

            print(f"[TranscriptionCoordinator] ✅ Pipeline completed successfully")
            return results

        except Exception as e:
            print(f"[TranscriptionCoordinator] ❌ Transcription processing failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def get_status(self):
        """Get coordinator status for debugging."""
        return {
            "initialized": self._initialized,
            "whisper_actor": self.whisper_actor is not None,
            "diarization_actor": self.diarization_actor is not None,
        }
