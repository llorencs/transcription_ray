"""
Standalone preprocessing service that can be run independently.
"""

import os
import asyncio
import uvloop
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import ray
from pathlib import Path

from src.services.preprocessing_actors import PreprocessingActor
from src.database.mongodb import MongoDB

# Set up asyncio event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Initialize FastAPI app
app = FastAPI(title="Audio Preprocessing Service", version="1.0.0")

# Global services
db = None
preprocessing_actor = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global db, preprocessing_actor

    # Connect to MongoDB
    mongodb_url = os.getenv(
        "MONGODB_URL",
        "mongodb://admin:password123@localhost:27017/transcription_db?authSource=admin",
    )
    db = MongoDB(mongodb_url)
    await db.connect()

    # Initialize Ray
    ray_address = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    # Initialize preprocessing actor
    preprocessing_actor = PreprocessingActor.remote()

    print("Preprocessing service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global db
    if db:
        await db.disconnect()
    if ray.is_initialized():
        ray.shutdown()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "preprocessing"}


@app.post("/preprocess/{task_id}")
async def preprocess_audio_task(
    task_id: str,
    separate_vocals: bool = True,
    reduce_noise: bool = True,
    normalize: bool = True,
    enhance_speech: bool = True,
):
    """Preprocess audio for a specific task."""
    try:
        # Get task from database
        task = await db.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if not task.get("file_id"):
            raise HTTPException(status_code=400, detail="Task has no associated file")

        # Update task status
        await db.update_task(task_id, {"status": "preprocessing"})

        # Get file path
        file_data, filename = await db.get_file(task["file_id"])
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")

        # Create temp file
        temp_dir = Path("/app/temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{task['file_id']}_{filename}"

        with open(temp_path, "wb") as f:
            f.write(file_data)

        # Preprocess audio
        processed_path = await preprocessing_actor.preprocess_audio.remote(
            str(temp_path),
            separate_vocals=separate_vocals,
            reduce_noise=reduce_noise,
            normalize=normalize,
            enhance_speech=enhance_speech,
        )

        # Store processed file
        with open(processed_path, "rb") as f:
            processed_data = f.read()

        processed_file_id = await db.store_file(
            processed_data,
            f"preprocessed_{filename}",
            {
                "original_file_id": task["file_id"],
                "preprocessing_settings": {
                    "separate_vocals": separate_vocals,
                    "reduce_noise": reduce_noise,
                    "normalize": normalize,
                    "enhance_speech": enhance_speech,
                },
            },
        )

        # Update task with processed file
        await db.update_task(
            task_id,
            {
                "processed_file_id": processed_file_id,
                "status": "preprocessing_completed",
            },
        )

        # Cleanup temp files
        try:
            temp_path.unlink(missing_ok=True)
            Path(processed_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to cleanup temp files: {e}")

        return {
            "task_id": task_id,
            "processed_file_id": processed_file_id,
            "status": "completed",
        }

    except Exception as e:
        await db.update_task(
            task_id, {"status": "preprocessing_failed", "error_message": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@app.post("/analyze/{task_id}")
async def analyze_audio_task(task_id: str):
    """Analyze audio content for a specific task."""
    try:
        # Get task from database
        task = await db.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if not task.get("file_id"):
            raise HTTPException(status_code=400, detail="Task has no associated file")

        # Get file path
        file_data, filename = await db.get_file(task["file_id"])
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")

        # Create temp file
        temp_dir = Path("/app/temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{task['file_id']}_{filename}"

        with open(temp_path, "wb") as f:
            f.write(file_data)

        # Analyze audio
        analysis = await preprocessing_actor.analyze_audio_content.remote(
            str(temp_path)
        )

        # Store analysis in task
        await db.update_task(task_id, {"audio_analysis": analysis})

        # Cleanup temp file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to cleanup temp file: {e}")

        return {"task_id": task_id, "analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/preprocess/file/{file_id}")
async def preprocess_file_direct(
    file_id: str,
    separate_vocals: bool = True,
    reduce_noise: bool = True,
    normalize: bool = True,
    enhance_speech: bool = True,
):
    """Preprocess a file directly without task context."""
    try:
        # Get file
        file_data, filename = await db.get_file(file_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")

        # Create temp file
        temp_dir = Path("/app/temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{file_id}_{filename}"

        with open(temp_path, "wb") as f:
            f.write(file_data)

        # Preprocess audio
        processed_path = await preprocessing_actor.preprocess_audio.remote(
            str(temp_path),
            separate_vocals=separate_vocals,
            reduce_noise=reduce_noise,
            normalize=normalize,
            enhance_speech=enhance_speech,
        )

        # Store processed file
        with open(processed_path, "rb") as f:
            processed_data = f.read()

        processed_file_id = await db.store_file(
            processed_data,
            f"preprocessed_{filename}",
            {
                "original_file_id": file_id,
                "preprocessing_settings": {
                    "separate_vocals": separate_vocals,
                    "reduce_noise": reduce_noise,
                    "normalize": normalize,
                    "enhance_speech": enhance_speech,
                },
            },
        )

        # Cleanup temp files
        try:
            temp_path.unlink(missing_ok=True)
            Path(processed_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to cleanup temp files: {e}")

        return {
            "original_file_id": file_id,
            "processed_file_id": processed_file_id,
            "status": "completed",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run the service
    uvicorn.run(
        "src.services.preprocessor_service:app",
        host="0.0.0.0",
        port=8001,
        loop="uvloop",
    )
