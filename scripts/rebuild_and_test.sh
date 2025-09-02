#!/bin/bash

set -e

echo "🚀 Rebuilding and Testing Advanced Transcription Service"
echo "=" * 60

# Step 1: Stop current services
echo "1. Stopping current services..."
docker compose down

# Step 2: Rebuild images with no cache
echo "2. Rebuilding Docker images (this may take several minutes)..."
docker compose build --no-cache --parallel

# Step 2.5: Verify build
echo "2.5. Verifying build..."
chmod +x scripts/verify_build.sh
if bash scripts/verify_build.sh; then
    echo "   ✅ Build verification passed"
else
    echo "   ❌ Build verification failed"
    exit 1
fi

# Step 3: Start services
echo "3. Starting services..."
docker compose up -d

# Step 4: Wait for services to be ready
echo "4. Waiting for services to initialize..."

# Wait for MongoDB
echo "   Waiting for MongoDB..."
for i in {1..30}; do
    if docker compose exec -T mongodb mongosh --quiet --eval "db.runCommand('ping')" > /dev/null 2>&1; then
        echo "   ✅ MongoDB ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ MongoDB failed to start"
        exit 1
    fi
    sleep 2
done

# Wait for Redis
echo "   Waiting for Redis..."
for i in {1..30}; do
    if docker compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "   ✅ Redis ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Redis failed to start"
        exit 1
    fi
    sleep 2
done

# Wait for Ray cluster
echo "   Waiting for Ray cluster..."
for i in {1..60}; do
    if curl -s http://localhost:8265/api/cluster_status > /dev/null 2>&1; then
        echo "   ✅ Ray cluster ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ⚠️ Ray cluster not responding (may still work)"
        break
    fi
    sleep 3
done

# Wait for API
echo "   Waiting for API service..."
for i in {1..60}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "   ✅ API ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ❌ API failed to start"
        docker compose logs api
        exit 1
    fi
    sleep 3
done

echo ""
echo "5. Running verification tests..."

# Test 1: Verify ML dependencies in Ray container
echo "   Testing ML dependencies in Ray container..."
if docker compose exec ray-head python /app/scripts/verify_ray_deps.py; then
    echo "   ✅ ML dependencies verified"
else
    echo "   ❌ ML dependencies missing"
    exit 1
fi

# Test 2: Debug Ray actors
echo "   Testing Ray actors..."
if bash scripts/debug_ray_container.sh; then
    echo "   ✅ Ray actors working"
else
    echo "   ❌ Ray actors failed"
    exit 1
fi

echo ""
echo "🎉 Rebuild and verification completed successfully!"
echo ""
echo "📋 Service Status:"
docker compose ps

echo ""
echo "🧪 Ready to test with audio file:"
echo "   make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"
echo ""
echo "📊 Monitor services:"
echo "   API: http://localhost:8080/docs"
echo "   Ray Dashboard: http://localhost:8265"
echo ""
echo "📝 View logs:"
echo "   docker compose logs -f [service-name]"