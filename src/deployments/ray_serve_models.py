"""
Ray Serve deployments for transcription models.
Compatible with serve deploy configuration.
"""

import ray
from ray import serve
import torch
import numpy as np
import librosa
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Pydantic models for requests
class TranscriptionRequest(BaseModel):
    audio_path: str
    model: str = "base"
    language: Optional[str] = None
    prompt: Optional[str] = None
    diarize: bool = False
    preprocess: bool = False
    gpu: bool = True


class LanguageDetectionRequest(BaseModel):
    audio_path: str
    detect_only: bool = True


# Create FastAPI app
fastapi_app = FastAPI(title="Transcription Service", version="1.0.0")


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 2, "memory": 8_000_000_000},
)
class TranscriptionService:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_cache_path = self.config.get("model_cache_path", "/app/models")
        self.default_model = self.config.get("default_whisper_model", "base")
        self.enable_diarization = self.config.get("enable_diarization", True)

        self.whisper_models = {}
        self.diarization_pipeline = None
        self._load_models()

    def _load_models(self):
        """Load ML models on startup."""
        try:
            # Load default Whisper model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            self.whisper_models[self.default_model] = WhisperModel(
                self.default_model,
                device=device,
                compute_type=compute_type,
                download_root=f"{self.model_cache_path}/whisper",
            )

            print(f"Loaded Whisper {self.default_model} model on {device}")

            # Load diarization pipeline if enabled
            if self.enable_diarization:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                if hf_token:
                    try:
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
                        )
                        self.diarization_pipeline.to(torch.device(device))
                        print(f"Loaded diarization pipeline on {device}")
                    except Exception as e:
                        print(f"Failed to load diarization pipeline: {e}")
                else:
                    print("No Hugging Face token provided, diarization disabled")

        except Exception as e:
            print(f"Failed to load models: {e}")
            raise

    def _get_whisper_model(self, model_size: str):
        """Get or load Whisper model."""
        if model_size not in self.whisper_models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            self.whisper_models[model_size] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=f"{self.model_cache_path}/whisper",
            )
            print(f"Loaded Whisper {model_size} model on {device}")

        return self.whisper_models[model_size]

    def transcribe_audio(self, request: TranscriptionRequest):
        """Main transcription endpoint."""
        try:
            # Get Whisper model
            model = self._get_whisper_model(request.model)

            # Transcribe with word timestamps
            segments, info = model.transcribe(
                request.audio_path,
                language=request.language if request.language != "auto" else None,
                initial_prompt=request.prompt,
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

            # Perform diarization if requested and available
            speaker_segments = []
            if (
                request.diarize
                and self.diarization_pipeline
                and self.enable_diarization
            ):
                try:
                    diarization = self.diarization_pipeline(request.audio_path)

                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        speaker_segments.append(
                            {"start": turn.start, "end": turn.end, "speaker": speaker}
                        )

                    # Assign speakers to words
                    words = self._assign_speakers(words, speaker_segments)

                    # Update segments with speaker info
                    for segment in result_segments:
                        segment["words"] = [
                            word
                            for word in words
                            if segment["start"] <= word["start"] <= segment["end"]
                        ]
                        # Assign majority speaker to segment
                        segment_speakers = {}
                        for word in segment["words"]:
                            if word.get("speaker"):
                                speaker = word["speaker"]
                                segment_speakers[speaker] = (
                                    segment_speakers.get(speaker, 0) + 1
                                )

                        if segment_speakers:
                            segment["speaker"] = max(
                                segment_speakers.items(), key=lambda x: x[1]
                            )[0]

                except Exception as e:
                    print(f"Diarization failed: {e}")

            return {
                "segments": result_segments,
                "words": words,
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": sum(s["end"] - s["start"] for s in result_segments),
                "speaker_segments": speaker_segments,
                "model": request.model,
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Transcription failed: {str(e)}"
            )

    def _assign_speakers(self, words, speaker_segments):
        """Assign speaker labels to words."""
        for word in words:
            word_center = (word["start"] + word["end"]) / 2
            for speaker_seg in speaker_segments:
                if speaker_seg["start"] <= word_center <= speaker_seg["end"]:
                    word["speaker"] = speaker_seg["speaker"]
                    break
        return words


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5, "num_cpus": 1, "memory": 4_000_000_000},
)
class LanguageDetectionService:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_cache_path = self.config.get("model_cache_path", "/app/models")
        self.whisper_model_size = self.config.get("whisper_model", "base")

        self.whisper_model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model for language detection."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            self.whisper_model = WhisperModel(
                self.whisper_model_size,
                device=device,
                compute_type=compute_type,
                download_root=f"{self.model_cache_path}/whisper",
            )

            print(
                f"Loaded Whisper {self.whisper_model_size} for language detection on {device}"
            )

        except Exception as e:
            print(f"Failed to load language detection model: {e}")
            raise

    def detect_language(self, request: LanguageDetectionRequest):
        """Language detection endpoint."""
        try:
            # Load first 30 seconds for language detection
            segments, info = self.whisper_model.transcribe(
                request.audio_path,
                language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Consume first segment to get language info
            first_segment = next(segments, None)

            return {"language": info.language, "confidence": info.language_probability}

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Language detection failed: {str(e)}"
            )


# FastAPI routes
@fastapi_app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    """HTTP endpoint for transcription."""
    transcription_service = serve.get_deployment("TranscriptionService").get_handle()
    return await transcription_service.transcribe_audio.remote(request)


@fastapi_app.post("/detect-language")
async def detect_language_endpoint(request: LanguageDetectionRequest):
    """HTTP endpoint for language detection."""
    language_service = serve.get_deployment("LanguageDetectionService").get_handle()
    return await language_service.detect_language.remote(request)


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "transcription-ray-serve"}


# For programmatic deployment (if needed) - REMOVED because we use YAML config
# The deployments are handled by serve deploy /app/config/serve_config.yaml
