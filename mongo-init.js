// MongoDB initialization script
db = db.getSiblingDB('transcription_db');

// Create collections
db.createCollection('files');
db.createCollection('tasks');
db.createCollection('results');

// Create indexes
db.files.createIndex({ "file_id": 1 }, { unique: true });
db.files.createIndex({ "created_at": 1 });
db.files.createIndex({ "metadata.content_type": 1 });

db.tasks.createIndex({ "task_id": 1 }, { unique: true });
db.tasks.createIndex({ "status": 1 });
db.tasks.createIndex({ "created_at": 1 });
db.tasks.createIndex({ "file_id": 1 });
db.tasks.createIndex({ "model": 1 });

db.results.createIndex({ "task_id": 1 }, { unique: true });
db.results.createIndex({ "created_at": 1 });

print('Database initialized successfully');