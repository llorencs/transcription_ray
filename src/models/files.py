"""
Model for files management.
This file maintains compatibility with the original files.py specification.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FileResponse(BaseModel):
    """Response model for file information"""

    id: str
    name: str
    suffix: Optional[str] = None


class FilesResponse(BaseModel):
    """Response model for listing files"""

    files: List[FileResponse]


# Extended models for enhanced functionality
class FileUploadResponse(BaseModel):
    """Enhanced response model for file upload"""

    id: str
    name: str
    suffix: Optional[str] = None
    size: int
    content_type: Optional[str] = None
    checksum: Optional[str] = None
    uploaded_at: datetime


class FileDetailResponse(BaseModel):
    """Detailed file information"""

    id: str
    name: str
    suffix: Optional[str] = None
    size: int
    content_type: Optional[str] = None
    checksum: Optional[str] = None
    uploaded_at: datetime
    last_accessed: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_status: Optional[str] = (
        None  # 'pending', 'processing', 'completed', 'failed'
    )


class FileSearchRequest(BaseModel):
    """Request model for searching files"""

    query: Optional[str] = None
    content_type: Optional[str] = None
    size_min: Optional[int] = None
    size_max: Optional[int] = None
    uploaded_after: Optional[datetime] = None
    uploaded_before: Optional[datetime] = None
    limit: int = Field(default=50, le=1000)
    offset: int = Field(default=0, ge=0)


class FileSearchResponse(BaseModel):
    """Response model for file search"""

    files: List[FileDetailResponse]
    total_count: int
    has_more: bool
    next_offset: Optional[int] = None


class FileBatchRequest(BaseModel):
    """Request model for batch file operations"""

    file_ids: List[str]
    operation: str  # 'delete', 'process', 'move', etc.
    parameters: Optional[Dict[str, Any]] = None


class FileBatchResponse(BaseModel):
    """Response model for batch operations"""

    batch_id: str
    operation: str
    total_files: int
    successful: int = 0
    failed: int = 0
    status: str  # 'pending', 'processing', 'completed', 'failed'
    errors: Optional[List[str]] = None


class FileValidationRequest(BaseModel):
    """Request model for file validation"""

    file_id: str
    validation_type: str  # 'audio', 'video', 'format', 'integrity'
    strict: bool = False


class FileValidationResponse(BaseModel):
    """Response model for file validation"""

    file_id: str
    valid: bool
    validation_type: str
    details: Dict[str, Any]
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class AudioFileInfo(BaseModel):
    """Audio file specific information"""

    duration: float  # in seconds
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    format: Optional[str] = None


class VideoFileInfo(BaseModel):
    """Video file specific information"""

    duration: float  # in seconds
    width: int
    height: int
    fps: float
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    format: Optional[str] = None
    audio_tracks: Optional[List[AudioFileInfo]] = None


class FileAnalysisRequest(BaseModel):
    """Request model for file analysis"""

    file_id: str
    analysis_type: str  # 'audio', 'video', 'content', 'quality'
    deep_analysis: bool = False


class FileAnalysisResponse(BaseModel):
    """Response model for file analysis"""

    file_id: str
    analysis_type: str
    audio_info: Optional[AudioFileInfo] = None
    video_info: Optional[VideoFileInfo] = None
    content_analysis: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    recommendations: Optional[List[str]] = None


class FileStatsResponse(BaseModel):
    """File system statistics"""

    total_files: int
    total_size: int  # in bytes
    storage_used: float  # percentage
    by_type: Dict[str, int]
    recent_uploads: int  # last 24 hours
    processing_queue: int


class FilePermission(BaseModel):
    """File permission model"""

    file_id: str
    user_id: str
    permissions: List[str]  # 'read', 'write', 'delete', 'share'
    expires_at: Optional[datetime] = None


class FileShareRequest(BaseModel):
    """Request model for sharing files"""

    file_id: str
    expires_in: Optional[int] = None  # seconds
    password: Optional[str] = None
    download_limit: Optional[int] = None


class FileShareResponse(BaseModel):
    """Response model for file sharing"""

    share_id: str
    file_id: str
    share_url: str
    expires_at: Optional[datetime] = None
    download_count: int = 0
    download_limit: Optional[int] = None


# Legacy compatibility
FileModel = FileResponse
FilesModel = FilesResponse


# Export all models
__all__ = [
    "FileResponse",
    "FilesResponse",
    "FileUploadResponse",
    "FileDetailResponse",
    "FileSearchRequest",
    "FileSearchResponse",
    "FileBatchRequest",
    "FileBatchResponse",
    "FileValidationRequest",
    "FileValidationResponse",
    "AudioFileInfo",
    "VideoFileInfo",
    "FileAnalysisRequest",
    "FileAnalysisResponse",
    "FileStatsResponse",
    "FilePermission",
    "FileShareRequest",
    "FileShareResponse",
    "FileModel",  # Legacy
    "FilesModel",  # Legacy
]
