# Advanced Transcription Service

A comprehensive, scalable transcription service with speaker diarization, voice activity detection, and advanced audio preprocessing capabilities.

## Features

- **Multiple ASR Models**: Support for Faster-Whisper (base, large-v3)
- **Speaker Diarization**: Using pyannote.audio for speaker identification
- **Voice Activity Detection**: Intelligent audio segmentation
- **Advanced Preprocessing**: 
  - Music/vocal separation using Demucs
  - Noise reduction with spectral analysis
  - Audio normalization and enhancement
- **Flexible Output Formats**: SRT, VTT, ASS, TTML, JSON, CSV, plain text
- **Distributed Processing**: Ray-based scalable architecture
- **GPU Support**: CUDA 12.8+ acceleration for all models
- **RESTful API**: FastAPI-based service with async support
- **File Management**: GridFS-based storage system
- **Task Management**: Async task processing with status tracking
- **Modern Stack**: Python 3.12, PyTorch 2.5, CUDA 12.8, cuDNN 9

## Architecture

The system is composed of several microservices:

- **API Service** (`api`): Main FastAPI application handling requests
- **Ray Head Node** (`ray-head`): Distributed computing coordinator
- **Preprocessing Service** (`preprocessor`): Audio enhancement pipeline
- **MongoDB**: Database for metadata and file storage
- **Redis**: Caching and Ray cluster coordination

## Quick Start

### Prerequisites

- Docker and Docker Compose V2
- NVIDIA GPU with CUDA 12.8+ support (optional, for GPU acceleration)
- At least 12GB RAM recommended (16GB+ for large models)
- 15GB+ disk space for models and dependencies
- Ubuntu 24.04 or compatible Linux distribution recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd advanced-transcription-service
```

2. Set up environment variables:
```bash
# Create .env file
cat > .env << EOF
MONGODB_URL=mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin
RAY_ADDRESS=ray://ray-head:10001
REDIS_URL=redis://redis:6379
HUGGINGFACE_TOKEN=your_token_here  # Optional, for pyannote models
EOF
```

3. Start the services:
```bash
# For CPU-only deployment
docker compose up -d

# For GPU deployment (requires nvidia-docker)
docker compose -f docker-compose.yml up -d
```

4. Wait for services to initialize (first run downloads models):
```bash
# Check service status
docker compose logs -f api
```

5. Access the API documentation:
```
http://localhost:8080/docs
```

## API Usage

### Upload a File

```bash
curl -X POST "http://localhost:8080/files/upload" \
  -F "file=@audio.wav"
```

### Start Transcription

```bash
curl -X POST "http://localhost:8080/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "model": "base",
    "language": "auto",
    "diarize": true,
    "preprocess": true,
    "gpu": true
  }'
```

### Check Task Status

```bash
curl "http://localhost:8080/tasks/{task-id}"
```

### Download Results

```bash
# JSON format
curl "http://localhost:8080/results/{task-id}/json"

# SRT subtitles
curl "http://localhost:8080/results/{task-id}/srt"

# VTT subtitles
curl "http://localhost:8080/results/{task-id}/vtt"

# Plain text
curl "http://localhost:8080/results/{task-id}/txt"
```

## Configuration

### Model Selection

Available Whisper models:
- `base`: Fast, good quality (39 languages)
- `large-v3`: Best quality, slower (99+ languages)

### Preprocessing Options

- **separate_vocals**: Remove background music (using Demucs)
- **reduce_noise**: Apply noise reduction
- **normalize**: Audio level normalization
- **enhance_speech**: Speech-specific enhancement

### Diarization

Speaker diarization can be enabled per request. It will:
- Identify different speakers
- Assign speaker labels to words/segments
- Include speaker information in all output formats

## Advanced Features

### Audio Segmentation

The service automatically handles large audio files by:
1. Using VAD to detect speech segments
2. Splitting long files into manageable chunks
3. Processing chunks in parallel
4. Reassembling results with correct timestamps

### Custom Segmentation

Multiple segmentation strategies available:
- **Balanced**: Equal word distribution
- **Speaker-aware**: Respects speaker boundaries  
- **Sentence-based**: Natural sentence breaks
- **Adaptive**: Optimal reading speed

### Subtitle Formats

Supported output formats:
- **SRT**: Standard subtitle format
- **VTT**: WebVTT for web players
- **ASS**: Advanced styling support
- **TTML**: W3C standard format
- **JSON**: Structured data with word-level timing
- **CSV**: Spreadsheet-friendly format

## Monitoring

### Ray Dashboard
Access Ray cluster monitoring at `http://localhost:8265`

### Service Health
```bash
curl http://localhost:8080/health
curl http://localhost:8001/health  # Preprocessing service
```

### Logs
```bash
docker-compose logs -f [service-name]
```

## Performance Optimization

### GPU Usage
- Enable GPU processing with `"gpu": true` in requests
- Models automatically use CUDA when available
- Mixed precision (float16) used for better performance

### Scaling
- Ray automatically distributes tasks across available resources
- Add more Ray worker nodes by scaling the `ray-head` service
- Preprocessing and transcription can run on separate GPUs

### Memory Management
- Large files are processed in chunks to manage memory usage
- Temporary files are automatically cleaned up
- GridFS handles large file storage efficiently

## Model Downloads

Models are automatically downloaded on first use:

- **Whisper models**: Stored in `/app/models/whisper`
- **Pyannote models**: Cached in `/app/models/pyannote`
- **Demucs models**: Downloaded to `/app/models/demucs`

First run may take 10-15 minutes depending on internet speed.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use CPU processing
2. **Model Download Fails**: Check internet connection and disk space
3. **GPU Not Detected**: Ensure nvidia-docker is properly installed
4. **Long Processing Times**: Consider using smaller model or enable preprocessing

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1
docker-compose logs -f
```

### Reset Database
```bash
docker compose down -v
docker compose up -d
```

## API Reference

### File Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/files/upload` | POST | Upload audio file |
| `/files` | GET | List uploaded files |
| `/files/{file_id}` | GET | Download file |
| `/files/{file_id}` | DELETE | Delete file |

### Transcription

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe` | POST | Start transcription |
| `/transcribe/url` | POST | Transcribe from URL |
| `/detect-language` | POST | Detect audio language |

### Task Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | List all tasks |
| `/tasks/{task_id}` | GET | Get task status |
| `/tasks/{task_id}` | DELETE | Cancel task |

### Results

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/results/{task_id}/json` | GET | JSON with word timing |
| `/results/{task_id}/asr` | GET | ASR event format |
| `/results/{task_id}/srt` | GET | SRT subtitles |
| `/results/{task_id}/vtt` | GET | WebVTT subtitles |
| `/results/{task_id}/txt` | GET | Plain text |

## Development

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.api.txt

# Start MongoDB and Redis
docker-compose up -d mongodb redis

# Set environment variables
export MONGODB_URL=mongodb://admin:password123@localhost:27017/transcription_db?authSource=admin
export RAY_ADDRESS=ray://localhost:10001

# Start Ray cluster locally
ray start --head --dashboard-host 0.0.0.0

# Run API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Testing

```bash
# Run tests
pytest tests/

# Test API endpoints
python scripts/test_api.py
```

### Custom Models

To add custom Whisper models:

1. Place model files in `/app/models/whisper/custom-model/`
2. Update `WhisperTranscriptionActor` to support the new model
3. Add model name to API validation

## Production Deployment

### Security Considerations

- Change default MongoDB credentials
- Use environment-specific secrets management
- Enable API authentication/authorization
- Configure firewall rules for internal services
- Use HTTPS in production

### Scaling for Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  ray-worker:
    build:
      context: .
      dockerfile: Dockerfile.ray
    deploy:
      replicas: 3
    environment:
      - RAY_ADDRESS=ray://ray-head:10001
    command: ray start --address=ray-head:6379 --num-gpus=1
```

### Monitoring and Logging

- Set up centralized logging (ELK stack, Fluentd)
- Monitor resource usage (Prometheus + Grafana)
- Set up alerting for failed tasks
- Track API response times and error rates

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

- Create issues on GitHub for bugs and feature requests
- Check existing issues and documentation first
- Provide detailed information including logs and system specs

## Changelog

### v1.0.0
- Initial release
- Basic transcription with Faster-Whisper
- Speaker diarization support
- Audio preprocessing pipeline
- Multiple output formats
- Ray-based distributed processing