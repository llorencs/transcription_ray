#!/bin/bash

# Advanced Transcription Service Startup Script

set -e

echo "🚀 Starting Advanced Transcription Service with Direct Processing..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available (new or legacy)
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
    echo "⚠️ Using legacy docker-compose command. Consider upgrading to Docker Compose V2."
else
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "For Docker Compose V2: Install Docker Desktop or 'docker-compose-plugin'"
    echo "For legacy version: Install 'docker-compose'"
    exit 1
fi

echo "✅ Using Docker Compose: $DOCKER_COMPOSE"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models_cache temp logs config

# Set permissions
chmod 755 models_cache temp logs config

# Check for GPU support
GPU_SUPPORT=""
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 NVIDIA GPU detected, enabling GPU support..."
    GPU_SUPPORT="--gpus all"
else
    echo "ℹ️ No NVIDIA GPU detected, running in CPU mode..."
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating default environment configuration..."
    cat > .env << EOF
MONGODB_URL=mongodb://admin:password123@mongodb:27017/transcription_db?authSource=admin
RAY_ADDRESS=ray://ray-head:10001
REDIS_URL=redis://redis:6379
PYTHONUNBUFFERED=1
RAY_DISABLE_IMPORT_WARNING=1

# Optional: Add your Hugging Face token for pyannote models
# HUGGINGFACE_TOKEN=your_token_here
EOF
    echo "✅ Created .env file with default settings"
fi

# Pull latest images
echo "📦 Pulling Docker images..."
$DOCKER_COMPOSE pull

# Start services
echo "🏗️ Starting services..."
if [ -n "$GPU_SUPPORT" ]; then
    echo "🎮 Starting with GPU support..."
    $DOCKER_COMPOSE up -d
else
    echo "💻 Starting in CPU-only mode..."
    $DOCKER_COMPOSE -f docker-compose.yml up -d
fi

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."

# Wait for MongoDB
echo "🍃 Waiting for MongoDB..."
for i in {1..30}; do
    if $DOCKER_COMPOSE exec -T mongodb mongosh --quiet --eval "db.runCommand('ping')" > /dev/null 2>&1; then
        echo "✅ MongoDB is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ MongoDB failed to start"
        exit 1
    fi
    sleep 2
done

# Wait for Redis
echo "🔴 Waiting for Redis..."
for i in {1..30}; do
    if $DOCKER_COMPOSE exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Redis failed to start"
        exit 1
    fi
    sleep 2
done

# Wait for Ray cluster (optional - may not be needed for direct processing)
echo "⚡ Checking Ray cluster (optional)..."
for i in {1..30}; do
    if curl -s http://localhost:8265/api/cluster_status > /dev/null 2>&1; then
        echo "✅ Ray cluster is ready"
        RAY_AVAILABLE=true
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️ Ray cluster not available (using direct processing mode)"
        RAY_AVAILABLE=false
        break
    fi
    sleep 3
done

# Wait for API service
echo "🌐 Waiting for API service..."
for i in {1..60}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ API service is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ API service failed to start"
        echo "📋 Checking API logs:"
        $DOCKER_COMPOSE logs api
        exit 1
    fi
    sleep 5
done

echo ""
echo "🎉 Advanced Transcription Service is now running!"
echo ""
echo "📊 Service URLs:"
echo "  • API Documentation: http://localhost:8080/docs"
echo "  • API Health Check: http://localhost:8080/health"
if [ "$RAY_AVAILABLE" = true ]; then
    echo "  • Ray Dashboard: http://localhost:8265"
fi
echo ""
echo "🏗️ Architecture:"
echo "  • API Container: FastAPI (lightweight, no ML dependencies)"
echo "  • Ray Workers: Whisper + PyTorch + all ML models"
echo "  • Distributed processing with Ray Actors"
echo "  • GPU acceleration in Ray workers"
echo ""
echo "🧪 Test the service:"
echo "  make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"
echo ""
echo "📝 View logs:"
echo "  $DOCKER_COMPOSE logs -f"
echo ""
echo "🛑 Stop services:"
echo "  $DOCKER_COMPOSE down"
echo ""

# Show service status
echo "📋 Current service status:"
$DOCKER_COMPOSE ps

echo ""
echo "🔍 Quick health checks:"

# Check API
API_HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo "failed")
if echo "$API_HEALTH" | grep -q "healthy"; then
    echo "  ✅ API Service: Healthy"
else
    echo "  ❌ API Service: Not responding"
fi

# Check MongoDB
if $DOCKER_COMPOSE exec -T mongodb mongosh --quiet --eval "db.runCommand('ping')" > /dev/null 2>&1; then
    echo "  ✅ MongoDB: Connected"
else
    echo "  ❌ MongoDB: Connection failed"
fi

# Check Redis
if $DOCKER_COMPOSE exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "  ✅ Redis: Connected"
else
    echo "  ❌ Redis: Connection failed"
fi

# Check Ray (optional)
if [ "$RAY_AVAILABLE" = true ]; then
    echo "  ✅ Ray Cluster: Available (optional)"
else
    echo "  ⚠️ Ray Cluster: Not available (using direct processing)"
fi

echo ""
echo "🎯 Ready for transcription!"
echo "   The service uses Ray Actors for distributed ML processing"
echo "   with a lightweight FastAPI frontend."