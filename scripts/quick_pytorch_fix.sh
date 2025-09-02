#!/bin/bash

echo "🚀 Quick PyTorch Fix for WSL CUDA Issues"
echo "=" * 50

# Stop services
echo "1. Stopping services..."
docker compose down

# Rebuild ray container with stable PyTorch
echo "2. Rebuilding Ray container with stable PyTorch 2.1.2..."
docker compose build --no-cache ray-head

# Start just the basic services
echo "3. Starting core services..."
docker compose up -d mongodb redis ray-head

# Wait for Ray to be ready
echo "4. Waiting for Ray cluster..."
for i in {1..30}; do
    if curl -s http://localhost:8265/api/cluster_status > /dev/null 2>&1; then
        echo "   ✅ Ray cluster ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Ray cluster timeout"
        exit 1
    fi
    sleep 3
done

# Test safe CUDA detection
echo "5. Testing safe CUDA detection..."
docker compose exec ray-head python /app/scripts/safe_cuda_test.py

# Test simple actors with safe CUDA
echo "6. Testing actors with safe CUDA..."
docker compose exec ray-head python /app/scripts/test_simple_actor.py

echo ""
echo "✅ Quick PyTorch fix completed!"
echo ""
echo "🧪 If tests passed, now start full services:"
echo "   docker compose up -d"
echo "   make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"