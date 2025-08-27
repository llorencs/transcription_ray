"""
Ray actors for distributed transcription processing.
"""

import os
import ray
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import tempfile
import uuid
from datetime import datetime

from src.models.pydantic_models import (
    WordModel,
    SegmentModel,
    JSONModel,
    ASRModel,
    EventModel,
)
from src.core.segmenter import Segmenter, SegmenterConfig


@ray.remote
class WhisperTranscriptionActor:
    """Ray actor for Whisper transcription."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                else:
                    device = "cpu"
                    compute_type = "int8"
            else:
                device = self.device
                compute_type = "float16" if device == "cuda" else "int8"

            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root="/app/models/whisper",
            )

            print(f"Loaded Whisper model {self.model_size} on {device}")

        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict], str]:
        """Transcribe audio file."""
        try:
            # Transcribe with word timestamps
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Convert to our format
            word_list = []
            segment_list = []
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
                        word_list.append(word_dict)
                        segment_words.append(word_dict)

                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text,
                    "words": segment_words,
                }
                segment_list.append(segment_dict)

            result = {
                "words": word_list,
                "segments": segment_list,
                "language": info.language,
                "language_probability": info.language_probability,
            }

            return result, full_text.strip()

        except Exception as e:
            print(f"Transcription failed: {e}")
            raise

    def detect_language(self, audio_path: str) -> Dict[str, Any]:
        """Detect language of audio file."""
        try:
            # Load first 30 seconds for language detection
            segments, info = self.model.transcribe(
                audio_path,
                language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Consume first segment to get language info
            first_segment = next(segments, None)

            return {"language": info.language, "confidence": info.language_probability}

        except Exception as e:
            print(f"Language detection failed: {e}")
            raise


@ray.remote
class DiarizationActor:
    """Ray actor for speaker diarization."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the diarization pipeline."""
        try:
            # Use Hugging Face token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
            )

            # Set device
            if self.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device)

            self.pipeline.to(device)

            print(f"Loaded diarization pipeline on {device}")

        except Exception as e:
            print(f"Failed to load diarization pipeline: {e}")
            raise

    def diarize(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> Dict[str, List]:
        """Perform speaker diarization."""
        try:
            # Run diarization
            with ProgressHook() as hook:
                diarization = self.pipeline(
                    audio_path, hook=hook, num_speakers=num_speakers
                )

            # Convert to our format
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append(
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                )

            return {"speaker_segments": speaker_segments}

        except Exception as e:
            print(f"Diarization failed: {e}")
            raise


@ray.remote
class VADActor:
    """Ray actor for Voice Activity Detection."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load VAD pipeline."""
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection", use_auth_token=hf_token
            )

            # Set device
            if self.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device)

            self.pipeline.to(device)

            print(f"Loaded VAD pipeline on {device}")

        except Exception as e:
            print(f"Failed to load VAD pipeline: {e}")
            raise

    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Detect voice activity segments."""
        try:
            vad = self.pipeline(audio_path)

            voice_segments = []
            for segment in vad.get_timeline():
                voice_segments.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.end - segment.start,
                    }
                )

            return voice_segments

        except Exception as e:
            print(f"VAD failed: {e}")
            raise

    def split_audio_by_vad(
        self, audio_path: str, max_segment_duration: float = 600.0
    ) -> List[str]:
        """Split audio file based on VAD with maximum segment duration."""
        try:
            # Get voice activity segments
            vad_segments = self.detect_voice_activity(audio_path)

            if not vad_segments:
                return [audio_path]

            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)

            # Group segments respecting max duration
            audio_segments = []
            current_start = 0
            current_duration = 0

            temp_dir = Path(tempfile.gettempdir()) / "vad_splits"
            temp_dir.mkdir(exist_ok=True)

            for i, segment in enumerate(vad_segments):
                segment_duration = segment["end"] - segment["start"]

                # If adding this segment would exceed max duration, save current segment
                if (
                    current_duration + segment_duration > max_segment_duration
                    and current_duration > 0
                ):
                    # Save current segment
                    segment_filename = temp_dir / f"segment_{len(audio_segments)}.wav"
                    start_sample = int(current_start * sr)
                    end_sample = int(vad_segments[i - 1]["end"] * sr)

                    sf.write(segment_filename, audio[start_sample:end_sample], sr)
                    audio_segments.append(str(segment_filename))

                    # Start new segment
                    current_start = segment["start"]
                    current_duration = segment_duration
                else:
                    current_duration += segment_duration

            # Save last segment if it exists
            if current_duration > 0:
                segment_filename = temp_dir / f"segment_{len(audio_segments)}.wav"
                start_sample = int(current_start * sr)

                sf.write(segment_filename, audio[start_sample:], sr)
                audio_segments.append(str(segment_filename))

            return audio_segments if audio_segments else [audio_path]

        except Exception as e:
            print(f"Audio splitting failed: {e}")
            raise


@ray.remote
class TranscriptionCoordinator:
    """Coordinator for orchestrating the full transcription pipeline."""

    def __init__(self):
        self.whisper_actor = None
        self.diarization_actor = None
        self.vad_actor = None

    def initialize_actors(self, model_size: str, use_gpu: bool = True):
        """Initialize required actors."""
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self.whisper_actor = WhisperTranscriptionActor.remote(model_size, device)
        self.diarization_actor = DiarizationActor.remote(device)
        self.vad_actor = VADActor.remote(device)

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
            # Initialize actors if not done
            if not self.whisper_actor:
                self.initialize_actors(model_size, use_gpu)

            results = {
                "transcription": None,
                "diarization": None,
                "segments_processed": [],
                "processing_info": {
                    "model_size": model_size,
                    "language": language,
                    "diarization_enabled": diarize,
                    "gpu_used": use_gpu,
                    "started_at": datetime.utcnow().isoformat(),
                },
            }

            # Check if audio needs splitting
            audio_info = librosa.get_duration(filename=audio_path)

            if audio_info > max_segment_duration:
                # Split audio using VAD
                audio_segments = await self.vad_actor.split_audio_by_vad.remote(
                    audio_path, max_segment_duration
                )
            else:
                audio_segments = [audio_path]

            # Process each segment
            all_words = []
            all_segments = []
            full_text = ""

            for i, segment_path in enumerate(audio_segments):
                print(f"Processing segment {i+1}/{len(audio_segments)}")

                # Transcribe segment
                transcription_result, segment_text = (
                    await self.whisper_actor.transcribe.remote(
                        segment_path, language, initial_prompt
                    )
                )

                # Adjust timestamps for segment offset if needed
                if i > 0:
                    # Calculate time offset for this segment
                    time_offset = sum(
                        [
                            librosa.get_duration(filename=seg_path)
                            for seg_path in audio_segments[:i]
                        ]
                    )

                    # Adjust word timestamps
                    for word in transcription_result["words"]:
                        word["start"] += time_offset
                        word["end"] += time_offset

                    # Adjust segment timestamps
                    for segment in transcription_result["segments"]:
                        segment["start"] += time_offset
                        segment["end"] += time_offset
                        for word in segment["words"]:
                            word["start"] += time_offset
                            word["end"] += time_offset

                all_words.extend(transcription_result["words"])
                all_segments.extend(transcription_result["segments"])
                full_text += segment_text + " "

                results["segments_processed"].append(
                    {
                        "segment_index": i,
                        "path": segment_path,
                        "duration": librosa.get_duration(filename=segment_path),
                        "words_count": len(transcription_result["words"]),
                        "language": transcription_result.get("language"),
                        "language_probability": transcription_result.get(
                            "language_probability"
                        ),
                    }
                )

            # Combine all results
            results["transcription"] = {
                "words": all_words,
                "segments": all_segments,
                "text": full_text.strip(),
                "language": transcription_result.get("language"),
                "language_probability": transcription_result.get(
                    "language_probability"
                ),
            }

            # Perform diarization if requested
            if diarize:
                print("Performing speaker diarization...")
                diarization_result = await self.diarization_actor.diarize.remote(
                    audio_path
                )
                results["diarization"] = diarization_result

                # Assign speakers to words/segments based on timestamps
                results["transcription"] = self._assign_speakers_to_transcription(
                    results["transcription"], diarization_result["speaker_segments"]
                )

            results["processing_info"]["completed_at"] = datetime.utcnow().isoformat()
            results["processing_info"]["total_duration"] = audio_info
            results["processing_info"]["segments_count"] = len(audio_segments)

            # Cleanup temporary segment files
            for segment_path in audio_segments:
                if segment_path != audio_path:  # Don't delete original file
                    try:
                        Path(segment_path).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"Failed to cleanup segment file {segment_path}: {e}")

            return results

        except Exception as e:
            print(f"Transcription processing failed: {e}")
            raise

    def _assign_speakers_to_transcription(
        self, transcription: Dict, speaker_segments: List[Dict]
    ) -> Dict:
        """Assign speaker labels to transcription words/segments."""
        try:
            # Create a mapping of time ranges to speakers
            speaker_timeline = []
            for seg in speaker_segments:
                speaker_timeline.append(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": seg["speaker"],
                    }
                )

            # Sort by start time
            speaker_timeline.sort(key=lambda x: x["start"])

            # Function to find speaker at given time
            def find_speaker_at_time(timestamp):
                for speaker_seg in speaker_timeline:
                    if speaker_seg["start"] <= timestamp <= speaker_seg["end"]:
                        return speaker_seg["speaker"]
                return None

            # Assign speakers to words
            for word in transcription["words"]:
                word_center = (word["start"] + word["end"]) / 2
                word["speaker"] = find_speaker_at_time(word_center)

            # Assign speakers to segments (majority speaker)
            for segment in transcription["segments"]:
                segment_speakers = {}
                total_duration = 0

                for word in segment["words"]:
                    if word.get("speaker"):
                        speaker = word["speaker"]
                        duration = word["end"] - word["start"]
                        segment_speakers[speaker] = (
                            segment_speakers.get(speaker, 0) + duration
                        )
                        total_duration += duration

                # Assign majority speaker to segment
                if segment_speakers:
                    segment["speaker"] = max(
                        segment_speakers.items(), key=lambda x: x[1]
                    )[0]
                else:
                    segment["speaker"] = None

            return transcription

        except Exception as e:
            print(f"Speaker assignment failed: {e}")
            return transcription
