"""
MongoDB database connection and operations.
"""

import motor.motor_asyncio
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorGridFSBucket,
)
from typing import Optional, Dict, Any, List
from datetime import datetime
import gridfs
from bson import ObjectId


class MongoDB:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.fs: Optional[AsyncIOMotorGridFSBucket] = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.connection_url)
            self.database = self.client.transcription_db
            self.fs = AsyncIOMotorGridFSBucket(self.database)

            # Test connection
            await self.client.admin.command("ping")
            print("Connected to MongoDB successfully")

            # Create indexes
            await self._create_indexes()

        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()

    async def _create_indexes(self):
        """Create necessary indexes."""
        try:
            # Files collection indexes
            await self.database.files.create_index("file_id", unique=True)
            await self.database.files.create_index("created_at")

            # Tasks collection indexes
            await self.database.tasks.create_index("task_id", unique=True)
            await self.database.tasks.create_index("status")
            await self.database.tasks.create_index("created_at")
            await self.database.tasks.create_index("file_id")

            # Results collection indexes
            await self.database.results.create_index("task_id", unique=True)
            await self.database.results.create_index("created_at")

            print("Database indexes created successfully")

        except Exception as e:
            print(f"Failed to create indexes: {e}")

    # File operations
    async def store_file(
        self, file_data: bytes, filename: str, metadata: Optional[Dict] = None
    ) -> str:
        """Store file in GridFS and return file ID."""
        try:
            file_metadata = {
                "filename": filename,
                "upload_date": datetime.utcnow(),
                **(metadata or {}),
            }

            file_id = await self.fs.upload_from_stream(
                filename, file_data, metadata=file_metadata
            )

            # Store file record
            file_record = {
                "file_id": str(file_id),
                "filename": filename,
                "size": len(file_data),
                "created_at": datetime.utcnow(),
                "metadata": file_metadata,
            }

            await self.database.files.insert_one(file_record)

            return str(file_id)

        except Exception as e:
            print(f"Failed to store file: {e}")
            raise

    async def get_file(self, file_id: str) -> Optional[tuple[bytes, str]]:
        """Get file data and filename by ID."""
        try:
            file_obj_id = ObjectId(file_id)
            grid_out = await self.fs.open_download_stream(file_obj_id)
            file_data = await grid_out.read()
            filename = grid_out.metadata.get("filename", f"file_{file_id}")

            return file_data, filename

        except Exception as e:
            print(f"Failed to get file {file_id}: {e}")
            return None

    async def delete_file(self, file_id: str) -> bool:
        """Delete file by ID."""
        try:
            file_obj_id = ObjectId(file_id)
            await self.fs.delete(file_obj_id)
            await self.database.files.delete_one({"file_id": file_id})

            return True

        except Exception as e:
            print(f"Failed to delete file {file_id}: {e}")
            return False

    async def list_files(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """List files with pagination."""
        try:
            cursor = (
                self.database.files.find(
                    {},
                    {"file_id": 1, "filename": 1, "size": 1, "created_at": 1, "_id": 0},
                )
                .skip(skip)
                .limit(limit)
                .sort("created_at", -1)
            )

            files = await cursor.to_list(length=limit)
            return files

        except Exception as e:
            print(f"Failed to list files: {e}")
            return []

    # Task operations
    async def create_task(self, task_data: Dict[str, Any]) -> str:
        """Create a new task."""
        try:
            task_data["created_at"] = datetime.utcnow()
            task_data["updated_at"] = datetime.utcnow()

            result = await self.database.tasks.insert_one(task_data)
            return str(result.inserted_id)

        except Exception as e:
            print(f"Failed to create task: {e}")
            raise

    async def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task by ID."""
        try:
            task = await self.database.tasks.find_one({"task_id": task_id}, {"_id": 0})
            return task

        except Exception as e:
            print(f"Failed to get task {task_id}: {e}")
            return None

    async def update_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """Update task."""
        try:
            update_data["updated_at"] = datetime.utcnow()

            result = await self.database.tasks.update_one(
                {"task_id": task_id}, {"$set": update_data}
            )

            return result.modified_count > 0

        except Exception as e:
            print(f"Failed to update task {task_id}: {e}")
            return False

    async def list_tasks(
        self, skip: int = 0, limit: int = 100, filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """List tasks with pagination and filtering."""
        try:
            query = filter_dict or {}

            cursor = (
                self.database.tasks.find(query, {"_id": 0})
                .skip(skip)
                .limit(limit)
                .sort("created_at", -1)
            )

            tasks = await cursor.to_list(length=limit)
            return tasks

        except Exception as e:
            print(f"Failed to list tasks: {e}")
            return []

    async def delete_task(self, task_id: str) -> bool:
        """Delete task by ID."""
        try:
            result = await self.database.tasks.delete_one({"task_id": task_id})
            return result.deleted_count > 0

        except Exception as e:
            print(f"Failed to delete task {task_id}: {e}")
            return False

    # Result operations
    async def store_result(self, task_id: str, result_data: Dict[str, Any]) -> bool:
        """Store transcription result."""
        try:
            result_record = {
                "task_id": task_id,
                "result_data": result_data,
                "created_at": datetime.utcnow(),
            }

            # Use upsert to replace existing result
            result = await self.database.results.replace_one(
                {"task_id": task_id}, result_record, upsert=True
            )

            return True

        except Exception as e:
            print(f"Failed to store result for task {task_id}: {e}")
            return False

    async def get_result(self, task_id: str) -> Optional[Dict]:
        """Get result by task ID."""
        try:
            result = await self.database.results.find_one(
                {"task_id": task_id}, {"_id": 0}
            )
            return result

        except Exception as e:
            print(f"Failed to get result for task {task_id}: {e}")
            return None

    async def delete_result(self, task_id: str) -> bool:
        """Delete result by task ID."""
        try:
            result = await self.database.results.delete_one({"task_id": task_id})
            return result.deleted_count > 0

        except Exception as e:
            print(f"Failed to delete result for task {task_id}: {e}")
            return False

    # Statistics operations
    async def get_task_stats(self) -> Dict[str, int]:
        """Get task statistics."""
        try:
            pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]

            cursor = self.database.tasks.aggregate(pipeline)
            stats = {doc["_id"]: doc["count"] async for doc in cursor}

            # Get total files count
            total_files = await self.database.files.count_documents({})
            stats["total_files"] = total_files

            return stats

        except Exception as e:
            print(f"Failed to get task stats: {e}")
            return {}

    # Cleanup operations
    async def cleanup_old_files(self, days: int = 30) -> int:
        """Clean up files older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Find old files
            old_files = await self.database.files.find(
                {"created_at": {"$lt": cutoff_date}}, {"file_id": 1}
            ).to_list(length=None)

            deleted_count = 0
            for file_doc in old_files:
                if await self.delete_file(file_doc["file_id"]):
                    deleted_count += 1

            return deleted_count

        except Exception as e:
            print(f"Failed to cleanup old files: {e}")
            return 0

    async def cleanup_old_tasks(self, days: int = 30) -> int:
        """Clean up completed/failed tasks older than specified days."""
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Delete old completed/failed tasks and their results
            old_tasks = await self.database.tasks.find(
                {
                    "created_at": {"$lt": cutoff_date},
                    "status": {"$in": ["completed", "failed", "cancelled"]},
                },
                {"task_id": 1},
            ).to_list(length=None)

            deleted_count = 0
            for task_doc in old_tasks:
                task_id = task_doc["task_id"]
                await self.delete_result(task_id)
                if await self.delete_task(task_id):
                    deleted_count += 1

            return deleted_count

        except Exception as e:
            print(f"Failed to cleanup old tasks: {e}")
            return 0
