"""
File management service.
"""

import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import UploadFile
import mimetypes

from src.database.mongodb import MongoDB
from src.models.files import FileResponse


class FileService:
    def __init__(self, db: MongoDB):
        self.db = db
        self.allowed_audio_types = {
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/flac",
            "audio/aac",
            "audio/ogg",
            "audio/mp4",
            "audio/x-m4a",
            "audio/webm",
            "audio/3gpp",
            "audio/amr",
        }
        self.allowed_video_types = {
            "video/mp4",
            "video/mpeg",
            "video/quicktime",
            "video/x-msvideo",
            "video/webm",
            "video/3gpp",
            "video/x-flv",
        }
        self.max_file_size = 500 * 1024 * 1024  # 500MB

    async def upload_file(self, file: UploadFile) -> str:
        """Upload and validate a file."""
        try:
            # Validate file type
            if not self._is_valid_file_type(file.content_type, file.filename):
                raise ValueError(f"Unsupported file type: {file.content_type}")

            # Read file content
            file_content = await file.read()

            # Validate file size
            if len(file_content) > self.max_file_size:
                raise ValueError(
                    f"File too large. Maximum size is {self.max_file_size / (1024*1024):.1f}MB"
                )

            if len(file_content) == 0:
                raise ValueError("Empty file")

            # Generate metadata
            metadata = {
                "content_type": file.content_type,
                "original_filename": file.filename,
                "size": len(file_content),
            }

            # Store file in database
            file_id = await self.db.store_file(
                file_content, file.filename or "unnamed", metadata
            )

            return file_id
        except Exception as e:
            raise Exception(f"File upload failed: {str(e)}")

    async def download_file(self, file_id: str) -> tuple[bytes, str]:
        """Download a file by ID."""
        try:
            file_data = await self.db.get_file(file_id)

            if not file_data:
                raise FileNotFoundError(f"File with ID {file_id} not found")

            return file_data
        except Exception as e:
            raise Exception(f"Failed to download file {file_id}: {str(e)}")

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file by ID."""
        try:
            return await self.db.delete_file(file_id)
        except Exception as e:
            print(f"Failed to delete file {file_id}: {e}")
            return False

    async def list_files(self, skip: int = 0, limit: int = 100) -> List[FileResponse]:
        """List files with pagination."""
        try:
            files = await self.db.list_files(skip=skip, limit=limit)

            return [
                FileResponse(
                    id=file_doc["file_id"],
                    name=file_doc["filename"],
                    suffix=(
                        Path(file_doc["filename"]).suffix
                        if file_doc["filename"]
                        else None
                    ),
                )
                for file_doc in files
            ]
        except Exception as e:
            print(f"Failed to list files: {e}")
            return []

    async def get_file_info(self, file_id: str) -> Optional[dict]:
        """Get file information."""
        try:
            task = await self.db.get_task(file_id)
            return task
        except Exception as e:
            print(f"Failed to get file info for {file_id}: {e}")
            return None

    def _is_valid_file_type(
        self, content_type: Optional[str], filename: Optional[str]
    ) -> bool:
        """Validate if file type is supported."""
        # Check content type
        if (
            content_type in self.allowed_audio_types
            or content_type in self.allowed_video_types
        ):
            return True

        # Fallback to filename extension
        if filename:
            guessed_type, _ = mimetypes.guess_type(filename)
            if (
                guessed_type in self.allowed_audio_types
                or guessed_type in self.allowed_video_types
            ):
                return True

        return False

    async def get_temp_file_path(self, file_id: str) -> str:
        """Get temporary file path for processing."""
        try:
            file_data, filename = await self.download_file(file_id)

            # Create temp directory if it doesn't exist
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(exist_ok=True)

            # Generate unique temporary filename
            temp_filename = f"{file_id}_{filename}"
            temp_path = temp_dir / temp_filename

            # Write file to temp location - using standard file operations instead of aiofiles
            with open(temp_path, "wb") as f:
                f.write(file_data)

            return str(temp_path)
        except Exception as e:
            raise Exception(f"Failed to create temp file for {file_id}: {str(e)}")

    async def cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file."""
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to cleanup temp file {temp_path}: {e}")

    async def validate_audio_file(self, file_id: str) -> dict:
        """Validate and get info about audio file - simplified version without librosa."""
        try:
            # Note: This is a simplified version that doesn't use librosa
            # For full audio analysis, this would be done in Ray workers
            file_data, filename = await self.download_file(file_id)

            info = {
                "size": len(file_data),
                "filename": filename,
                "extension": Path(filename).suffix if filename else None,
                "note": "Detailed audio analysis performed in processing workers",
            }

            return info
        except Exception as e:
            raise ValueError(f"Failed to validate audio file {file_id}: {str(e)}")
