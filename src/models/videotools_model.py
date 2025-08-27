"""
Models for the Whisper transcription service.
This file maintains compatibility with the original videotools_model.py specification.
"""

from typing import Optional, Any
from pydantic import BaseModel


class TranscriptionReqModel(BaseModel):
    """Model for the transcription request"""

    model: str = "base"
    gpu: bool = False
    file_id: str
    language: Optional[str] = "auto"
    prompt: Optional[str] = None
    preprocess: Optional[bool] = False
    callback_url: Optional[str] = None
    diarize: Optional[bool] = False
    asr_format: Optional[bool] = False


class TranscriptionRespModel(TranscriptionReqModel):
    """Response model for the transcription request"""

    id: str
    status: str
    message: Optional[str] = None


class TaskRespModel(BaseModel):
    """Model for the task response"""

    id: str
    status: str
    result: Optional[Any] = None
    error_message: Optional[str] = None


class TransRespModel(BaseModel):
    """Model for the transcription response"""

    file_id: str
    language_declared: str
    language_detected: Optional[str] = None
    srt_file_id: str
    txt_file_id: str
    json_file_id: str
    asr_file_id: Optional[str] = None
    model: str
    text: Optional[str]


class TasksRespModel(BaseModel):
    """Model for the tasks response"""

    tasks: list[TaskRespModel]


class TasksRespIdsModel(BaseModel):
    """Model for the tasks response with IDs only"""

    tasks: list[str]


class languageDetectionModel(BaseModel):
    """Model for language detection response"""

    file_id: str
    language: str
    confidence: Optional[float] = None


class TranscriptionURLReqModel(BaseModel):
    """Model for the transcription request with URL"""

    url: str
    model: str = "base"
    gpu: bool = False
    language: Optional[str] = "auto"
    prompt: Optional[str] = None
    preprocess: Optional[bool] = False
    callback_url: Optional[str] = None
    diarize: Optional[bool] = False
    asr_format: Optional[bool] = False


# Extended models for additional functionality
class BatchTranscriptionReqModel(BaseModel):
    """Model for batch transcription requests"""

    file_ids: list[str]
    model: str = "base"
    gpu: bool = False
    language: Optional[str] = "auto"
    prompt: Optional[str] = None
    preprocess: Optional[bool] = False
    callback_url: Optional[str] = None
    diarize: Optional[bool] = False
    asr_format: Optional[bool] = False
    priority: int = 0


class BatchTranscriptionRespModel(BaseModel):
    """Response model for batch transcription"""

    batch_id: str
    status: str
    total_files: int
    completed_files: int = 0
    failed_files: int = 0
    message: Optional[str] = None


class PreprocessingReqModel(BaseModel):
    """Model for preprocessing request"""

    file_id: str
    separate_vocals: bool = True
    reduce_noise: bool = True
    normalize: bool = True
    enhance_speech: bool = True


class PreprocessingRespModel(BaseModel):
    """Response model for preprocessing"""

    task_id: str
    original_file_id: str
    processed_file_id: Optional[str] = None
    status: str
    message: Optional[str] = None


class DiarizationReqModel(BaseModel):
    """Model for diarization request"""

    file_id: str
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


class DiarizationRespModel(BaseModel):
    """Response model for diarization"""

    task_id: str
    file_id: str
    status: str
    speakers_detected: Optional[int] = None
    message: Optional[str] = None


class VADReqModel(BaseModel):
    """Model for Voice Activity Detection request"""

    file_id: str
    min_duration: float = 0.5
    max_pause: float = 2.0


class VADRespModel(BaseModel):
    """Response model for VAD"""

    task_id: str
    file_id: str
    status: str
    voice_segments: Optional[int] = None
    total_voice_duration: Optional[float] = None
    message: Optional[str] = None


class ModelInfoModel(BaseModel):
    """Model information"""

    name: str
    size: str
    languages: list[str]
    gpu_required: bool = False
    memory_requirement: str
    description: Optional[str] = None


class ServiceInfoModel(BaseModel):
    """Service information model"""

    service_name: str
    version: str
    available_models: list[ModelInfoModel]
    supported_formats: list[str]
    max_file_size: int
    gpu_available: bool
    features: dict[str, bool]


class QuotaModel(BaseModel):
    """User quota model"""

    user_id: str
    requests_remaining: int
    requests_total: int
    reset_time: str
    premium: bool = False


class WebhookModel(BaseModel):
    """Webhook configuration model"""

    url: str
    events: list[str]  # ['completed', 'failed', 'started']
    secret: Optional[str] = None
    active: bool = True


class CallbackDataModel(BaseModel):
    """Data sent to callback URLs"""

    task_id: str
    status: str
    timestamp: str
    event_type: str
    data: Optional[dict] = None
    error: Optional[str] = None


# Legacy compatibility models - keeping original names
class TranscriptionReqModel_Legacy(TranscriptionReqModel):
    """Legacy model name for backward compatibility"""

    pass


class TranscriptionRespModel_Legacy(TranscriptionRespModel):
    """Legacy model name for backward compatibility"""

    pass


# Export all models for easy imports
__all__ = [
    "TranscriptionReqModel",
    "TranscriptionRespModel",
    "TaskRespModel",
    "TransRespModel",
    "TasksRespModel",
    "TasksRespIdsModel",
    "languageDetectionModel",
    "TranscriptionURLReqModel",
    "BatchTranscriptionReqModel",
    "BatchTranscriptionRespModel",
    "PreprocessingReqModel",
    "PreprocessingRespModel",
    "DiarizationReqModel",
    "DiarizationRespModel",
    "VADReqModel",
    "VADRespModel",
    "ModelInfoModel",
    "ServiceInfoModel",
    "QuotaModel",
    "WebhookModel",
    "CallbackDataModel",
    "TranscriptionReqModel_Legacy",
    "TranscriptionRespModel_Legacy",
]
