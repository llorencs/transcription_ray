"""
Main FastAPI application for the transcription service.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, List
import os
import asyncio
import io
from pathlib import Path

from src.models.videotools_model import (
    TranscriptionReqModel,
    TranscriptionRespModel,
    TaskRespModel,
    TasksRespModel,
    TranscriptionURLReqModel,
    languageDetectionModel,
)
from src.models.files import FileResponse, FilesResponse
from src.models.pydantic_models import ASRModel, JSONModel
from src.database.mongodb import MongoDB
from src.services.file_service import FileService
from src.utils.ray_client import RayClient


# Simple transcription service for initial testing
class SimpleTranscriptionService:
    def __init__(self, db: MongoDB, ray_client: RayClient):
        self.db = db
        self.ray_client = ray_client

    async def start_transcription(self, request: TranscriptionReqModel) -> str:
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "file_id": request.file_id,
            "model": request.model,
            "status": "pending",
            "created_at": datetime.utcnow(),
        }
        await self.db.create_task(task_data)
        return task_id

    async def start_transcription_from_url(
        self, request: TranscriptionURLReqModel
    ) -> str:
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "source_url": request.url,
            "model": request.model,
            "status": "pending",
            "created_at": datetime.utcnow(),
        }
        await self.db.create_task(task_data)
        return task_id

    async def get_task(self, task_id: str) -> Optional[TaskRespModel]:
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
        return await self.db.update_task(task_id, {"status": "cancelled"})

    async def get_json_result(self, task_id: str) -> Optional[JSONModel]:
        # Placeholder - return None for now
        return None

    async def get_asr_result(self, task_id: str) -> Optional[ASRModel]:
        # Placeholder - return None for now
        return None

    async def get_srt_result(self, task_id: str) -> Optional[str]:
        # Placeholder - return None for now
        return None

    async def get_vtt_result(self, task_id: str) -> Optional[str]:
        # Placeholder - return None for now
        return None

    async def get_txt_result(self, task_id: str) -> Optional[str]:
        # Placeholder - return None for now
        return None

    async def detect_language(self, file_id: str) -> languageDetectionModel:
        return languageDetectionModel(file_id=file_id, language="en", confidence=0.9)


import uuid
from datetime import datetime

app = FastAPI(
    title="Advanced Transcription Service",
    description="A comprehensive transcription service with diarization, VAD, and preprocessing",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
db = None
file_service = None
transcription_service = None
ray_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global db, file_service, transcription_service, ray_client

    # Initialize MongoDB
    mongodb_url = os.getenv(
        "MONGODB_URL",
        "mongodb://admin:password123@localhost:27017/transcription_db?authSource=admin",
    )
    db = MongoDB(mongodb_url)
    await db.connect()

    # Initialize services
    file_service = FileService(db)
    ray_client = RayClient()
    await ray_client.connect()
    transcription_service = SimpleTranscriptionService(db, ray_client)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global db, ray_client
    if db:
        await db.disconnect()
    if ray_client:
        await ray_client.disconnect()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "transcription-api"}


# File management endpoints
@app.post("/files/upload", response_model=FileResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return its ID."""
    try:
        file_id = await file_service.upload_file(file)
        return FileResponse(
            id=file_id,
            name=file.filename,
            suffix=Path(file.filename).suffix if file.filename else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/files", response_model=FilesResponse)
async def list_files(skip: int = 0, limit: int = 100):
    """List uploaded files."""
    try:
        files = await file_service.list_files(skip=skip, limit=limit)
        return FilesResponse(files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.get("/files/{file_id}")
async def download_file(file_id: str):
    """Download a file by ID."""
    try:
        file_data, filename = await file_service.download_file(file_id)
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file by ID."""
    try:
        success = await file_service.delete_file(file_id)
        if success:
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


# Language detection endpoint
@app.post("/detect-language", response_model=languageDetectionModel)
async def detect_language(file_id: str):
    """Detect the language of an audio file."""
    try:
        result = await transcription_service.detect_language(file_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Language detection failed: {str(e)}"
        )


# Transcription endpoints
@app.post("/transcribe", response_model=TranscriptionRespModel)
async def transcribe_audio(request: TranscriptionReqModel):
    """Start a transcription task."""
    try:
        task_id = await transcription_service.start_transcription(request)
        return TranscriptionRespModel(
            **request.dict(),
            id=task_id,
            status="pending",
            message="Transcription task started",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Transcription failed to start: {str(e)}"
        )


@app.post("/transcribe/url", response_model=TranscriptionRespModel)
async def transcribe_from_url(request: TranscriptionURLReqModel):
    """Start a transcription task from URL."""
    try:
        task_id = await transcription_service.start_transcription_from_url(request)
        return TranscriptionRespModel(
            model=request.model,
            gpu=request.gpu,
            file_id="",  # Will be filled when URL is downloaded
            language=request.language,
            prompt=request.prompt,
            preprocess=request.preprocess,
            callback_url=request.callback_url,
            diarize=request.diarize,
            asr_format=request.asr_format,
            id=task_id,
            status="pending",
            message="Transcription task started from URL",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"URL transcription failed to start: {str(e)}"
        )


# Task management endpoints
@app.get("/tasks", response_model=TasksRespModel)
async def list_tasks(skip: int = 0, limit: int = 100):
    """List all tasks."""
    try:
        tasks = await transcription_service.list_tasks(skip=skip, limit=limit)
        return TasksRespModel(tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@app.get("/tasks/{task_id}", response_model=TaskRespModel)
async def get_task(task_id: str):
    """Get task status and result."""
    try:
        task = await transcription_service.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task: {str(e)}")


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task."""
    try:
        success = await transcription_service.cancel_task(task_id)
        if success:
            return {"message": "Task cancelled successfully"}
        else:
            raise HTTPException(
                status_code=404, detail="Task not found or cannot be cancelled"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


# Result endpoints
@app.get("/results/{task_id}/json", response_model=JSONModel)
async def get_json_result(task_id: str):
    """Get transcription result in JSON format."""
    try:
        result = await transcription_service.get_json_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get JSON result: {str(e)}"
        )


@app.get("/results/{task_id}/asr", response_model=ASRModel)
async def get_asr_result(task_id: str):
    """Get transcription result in ASR format."""
    try:
        result = await transcription_service.get_asr_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get ASR result: {str(e)}"
        )


@app.get("/results/{task_id}/srt")
async def get_srt_result(task_id: str):
    """Get transcription result in SRT format."""
    try:
        srt_content = await transcription_service.get_srt_result(task_id)
        if not srt_content:
            raise HTTPException(status_code=404, detail="Result not found")

        return StreamingResponse(
            io.StringIO(srt_content),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={task_id}.srt"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get SRT result: {str(e)}"
        )


@app.get("/results/{task_id}/vtt")
async def get_vtt_result(task_id: str):
    """Get transcription result in VTT format."""
    try:
        vtt_content = await transcription_service.get_vtt_result(task_id)
        if not vtt_content:
            raise HTTPException(status_code=404, detail="Result not found")

        return StreamingResponse(
            io.StringIO(vtt_content),
            media_type="text/vtt",
            headers={"Content-Disposition": f"attachment; filename={task_id}.vtt"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get VTT result: {str(e)}"
        )


@app.get("/results/{task_id}/txt")
async def get_txt_result(task_id: str):
    """Get transcription result in plain text format."""
    try:
        txt_content = await transcription_service.get_txt_result(task_id)
        if not txt_content:
            raise HTTPException(status_code=404, detail="Result not found")

        return StreamingResponse(
            io.StringIO(txt_content),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={task_id}.txt"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get TXT result: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
