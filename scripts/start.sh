#!/bin/bash

# Advanced Transcription Service Startup Script

set -e

echo "🚀 Starting Advanced Transcription Service..."

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
    # Remove GPU-specific configuration for CPU-only deployment
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

# Wait for Ray
echo "⚡ Waiting for Ray cluster..."
for i in {1..60}; do
    if curl -s http://localhost:8265/api/cluster_status > /dev/null 2>&1; then
        echo "✅ Ray cluster is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Ray cluster failed to start"
        exit 1
    fi
    sleep 5
done

# Wait for API
echo "🌐 Waiting for API service..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ API service is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API service failed to start"
        exit 1
    fi
    sleep 5
done

# Deploy Ray Serve models
echo "🤖 Deploying ML models..."
$DOCKER_COMPOSE exec -T ray-head python src/deployments/ray_serve_models.py || echo "⚠️ Model deployment may take a few minutes on first run"

echo ""
echo "🎉 Advanced Transcription Service is now running!"
echo ""
echo "📊 Service URLs:"
echo "  • API Documentation: http://localhost:8080/docs"
echo "  • API Health Check: http://localhost:8080/health"
echo "  • Ray Dashboard: http://localhost:8265"
echo ""
echo "🧪 Test the service:"
echo "  python scripts/test_api.py --audio-file /path/to/audio.wav"
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