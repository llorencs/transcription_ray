from pydantic import BaseModel, Field
from typing import Optional, List, Annotated


class WordModel(BaseModel):
    """Model for a word with timestamps"""

    start: Annotated[float, Field(..., description="Start time of the word in seconds")]
    end: Annotated[float, Field(..., description="End time of the word in seconds")]
    text: Annotated[str, Field(..., description="Transcribed text of the word")]
    confidence: Annotated[
        Optional[float],
        Field(None, description="Confidence score of the word transcription"),
    ]
    speaker: Annotated[
        Optional[str], Field(None, description="Speaker ID for the word, if applicable")
    ]


class SegmentModel(BaseModel):
    """Model for a segment with timestamps"""

    start: Annotated[
        float, Field(..., description="Start time of the segment in seconds")
    ]
    end: Annotated[float, Field(..., description="End time of the segment in seconds")]
    text: Annotated[str, Field(..., description="Transcribed text of the segment")]
    words: Annotated[
        List[WordModel], Field(..., description="List of words in the segment")
    ]
    speaker: Annotated[
        Optional[str],
        Field(None, description="Primary speaker for the segment, if applicable"),
    ]


class JSONModel(BaseModel):
    """Model for the complete transcription in JSON format"""

    text: Annotated[str, Field(..., description="Full transcribed text")]
    segments: Annotated[
        List[SegmentModel], Field(..., description="List of segments with timestamps")
    ]
    language: Annotated[
        Optional[str], Field(None, description="Detected or specified language")
    ]
    language_probability: Annotated[
        Optional[float], Field(None, description="Confidence of language detection")
    ]


class EventModel(BaseModel):
    """Model for an event in the ASR request"""

    content: Annotated[str, Field(..., description="Content of the event")]
    start_time: Annotated[
        float, Field(..., description="Start time of the event in seconds")
    ]
    end_time: Annotated[
        float, Field(..., description="End time of the event in seconds")
    ]
    event_type: Annotated[
        str, Field(..., description="Type of the event (e.g., 'word', 'punctuation')")
    ]
    language: Annotated[
        Optional[str], Field(None, description="Language of the event, if applicable")
    ]
    confidence: Annotated[
        Optional[float],
        Field(None, description="Confidence score of the event, if applicable"),
    ]
    speaker: Annotated[
        Optional[str],
        Field(None, description="Speaker ID for the event, if applicable"),
    ]
    is_eol: Annotated[
        Optional[bool], Field(False, description="Whether this event marks end of line")
    ]
    is_eos: Annotated[
        Optional[bool],
        Field(False, description="Whether this event marks end of segment"),
    ]


class ASRModel(BaseModel):
    """Model for the ASR request with events"""

    asr_model: Annotated[
        str, Field(..., description="ASR model used for transcription")
    ]
    created_at: Annotated[str, Field(..., description="Creation timestamp")]
    generated_by: Annotated[
        str, Field(..., description="Service that generated the ASR")
    ]
    version: Annotated[int, Field(default=1, description="Version of the ASR model")]
    events: Annotated[
        List[EventModel], Field(..., description="List of events in the ASR")
    ]
    language: Annotated[
        Optional[str], Field(None, description="Detected or specified language")
    ]
    language_probability: Annotated[
        Optional[float], Field(None, description="Confidence of language detection")
    ]
    duration: Annotated[
        Optional[float],
        Field(None, description="Total duration of the audio in seconds"),
    ]
    processing_info: Annotated[
        Optional[dict], Field(None, description="Additional processing information")
    ]


class DiarizationModel(BaseModel):
    """Model for speaker diarization results"""

    speakers: Annotated[
        List[str], Field(..., description="List of identified speaker IDs")
    ]
    speaker_segments: Annotated[
        List[dict], Field(..., description="List of speaker segments with timestamps")
    ]
    total_speakers: Annotated[
        int, Field(..., description="Total number of speakers detected")
    ]


class VADModel(BaseModel):
    """Model for Voice Activity Detection results"""

    voice_segments: Annotated[
        List[dict], Field(..., description="List of voice activity segments")
    ]
    total_voice_duration: Annotated[
        float, Field(..., description="Total duration of voice activity in seconds")
    ]
    voice_activity_ratio: Annotated[
        float, Field(..., description="Ratio of voice activity to total duration")
    ]


class PreprocessingModel(BaseModel):
    """Model for audio preprocessing results"""

    original_file_id: Annotated[
        str, Field(..., description="ID of the original audio file")
    ]
    processed_file_id: Annotated[
        str, Field(..., description="ID of the processed audio file")
    ]
    preprocessing_steps: Annotated[
        List[str], Field(..., description="List of preprocessing steps applied")
    ]
    settings: Annotated[dict, Field(..., description="Settings used for preprocessing")]
    quality_metrics: Annotated[
        Optional[dict],
        Field(None, description="Quality metrics before/after preprocessing"),
    ]


class SubtitleModel(BaseModel):
    """Model for subtitle data"""

    format: Annotated[
        str, Field(..., description="Subtitle format (srt, vtt, ass, etc.)")
    ]
    content: Annotated[
        str, Field(..., description="Subtitle content in the specified format")
    ]
    segments: Annotated[
        List[SegmentModel], Field(..., description="Source segments used for subtitles")
    ]
    metadata: Annotated[
        Optional[dict], Field(None, description="Additional subtitle metadata")
    ]


class TranscriptionResultModel(BaseModel):
    """Comprehensive model for complete transcription results"""

    task_id: Annotated[str, Field(..., description="Task ID")]
    file_id: Annotated[str, Field(..., description="Source file ID")]
    status: Annotated[str, Field(..., description="Processing status")]

    # Transcription data
    transcription: Annotated[
        Optional[JSONModel], Field(None, description="JSON transcription result")
    ]
    asr_result: Annotated[
        Optional[ASRModel], Field(None, description="ASR format result")
    ]

    # Optional results
    diarization: Annotated[
        Optional[DiarizationModel],
        Field(None, description="Speaker diarization results"),
    ]
    vad_result: Annotated[
        Optional[VADModel], Field(None, description="Voice activity detection results")
    ]
    preprocessing: Annotated[
        Optional[PreprocessingModel], Field(None, description="Preprocessing results")
    ]

    # Subtitle formats
    subtitles: Annotated[
        Optional[dict], Field(None, description="Available subtitle formats")
    ]

    # Metadata
    processing_time: Annotated[
        Optional[float], Field(None, description="Total processing time in seconds")
    ]
    model_info: Annotated[
        Optional[dict], Field(None, description="Model information used")
    ]
    quality_metrics: Annotated[
        Optional[dict], Field(None, description="Quality assessment metrics")
    ]
    created_at: Annotated[Optional[str], Field(None, description="Creation timestamp")]
    completed_at: Annotated[
        Optional[str], Field(None, description="Completion timestamp")
    ]


class BatchTranscriptionModel(BaseModel):
    """Model for batch transcription requests"""

    batch_id: Annotated[str, Field(..., description="Batch ID")]
    file_ids: Annotated[
        List[str], Field(..., description="List of file IDs to process")
    ]
    model: Annotated[str, Field("base", description="Whisper model to use")]
    language: Annotated[str, Field("auto", description="Language for transcription")]
    diarize: Annotated[bool, Field(False, description="Enable speaker diarization")]
    preprocess: Annotated[bool, Field(False, description="Enable audio preprocessing")]
    gpu: Annotated[bool, Field(True, description="Use GPU if available")]
    callback_url: Annotated[
        Optional[str],
        Field(None, description="Webhook URL for completion notification"),
    ]
    priority: Annotated[
        int, Field(0, description="Processing priority (higher = more priority)")
    ]


class BatchResultModel(BaseModel):
    """Model for batch transcription results"""

    batch_id: Annotated[str, Field(..., description="Batch ID")]
    status: Annotated[str, Field(..., description="Batch status")]
    total_files: Annotated[int, Field(..., description="Total number of files")]
    completed_files: Annotated[int, Field(..., description="Number of completed files")]
    failed_files: Annotated[int, Field(..., description="Number of failed files")]
    results: Annotated[
        List[TranscriptionResultModel],
        Field(..., description="Individual file results"),
    ]
    created_at: Annotated[str, Field(..., description="Batch creation timestamp")]
    completed_at: Annotated[
        Optional[str], Field(None, description="Batch completion timestamp")
    ]


class HealthCheckModel(BaseModel):
    """Model for service health check"""

    status: Annotated[str, Field(..., description="Health status")]
    service: Annotated[str, Field(..., description="Service name")]
    version: Annotated[str, Field("1.0.0", description="Service version")]
    timestamp: Annotated[str, Field(..., description="Health check timestamp")]
    dependencies: Annotated[
        Optional[dict], Field(None, description="Status of service dependencies")
    ]
    resources: Annotated[
        Optional[dict], Field(None, description="Resource utilization information")
    ]


class ErrorModel(BaseModel):
    """Model for error responses"""

    error_code: Annotated[str, Field(..., description="Error code")]
    message: Annotated[str, Field(..., description="Error message")]
    details: Annotated[
        Optional[dict], Field(None, description="Additional error details")
    ]
    timestamp: Annotated[str, Field(..., description="Error timestamp")]
    trace_id: Annotated[
        Optional[str], Field(None, description="Trace ID for debugging")
    ]


class MetricsModel(BaseModel):
    """Model for service metrics"""

    total_tasks: Annotated[int, Field(..., description="Total number of tasks")]
    completed_tasks: Annotated[int, Field(..., description="Number of completed tasks")]
    failed_tasks: Annotated[int, Field(..., description="Number of failed tasks")]
    pending_tasks: Annotated[int, Field(..., description="Number of pending tasks")]
    average_processing_time: Annotated[
        float, Field(..., description="Average processing time in seconds")
    ]
    total_files_processed: Annotated[
        int, Field(..., description="Total number of files processed")
    ]
    total_audio_duration: Annotated[
        float, Field(..., description="Total duration of audio processed in seconds")
    ]
    gpu_utilization: Annotated[
        Optional[float], Field(None, description="GPU utilization percentage")
    ]
    memory_usage: Annotated[
        Optional[dict], Field(None, description="Memory usage statistics")
    ]
    active_workers: Annotated[
        Optional[int], Field(None, description="Number of active Ray workers")
    ]


class ConfigurationModel(BaseModel):
    """Model for service configuration"""

    whisper_models: Annotated[
        List[str], Field(..., description="Available Whisper models")
    ]
    supported_languages: Annotated[
        List[str], Field(..., description="Supported languages")
    ]
    max_file_size: Annotated[int, Field(..., description="Maximum file size in bytes")]
    supported_formats: Annotated[
        List[str], Field(..., description="Supported audio formats")
    ]
    subtitle_formats: Annotated[
        List[str], Field(..., description="Available subtitle formats")
    ]
    gpu_available: Annotated[
        bool, Field(..., description="Whether GPU processing is available")
    ]
    ray_cluster_info: Annotated[
        Optional[dict], Field(None, description="Ray cluster information")
    ]
