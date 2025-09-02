#!/bin/bash

set -e

echo "🚀 Fix CUDA Double Free Error in Ray Actors"
echo "=" * 60

echo ""
echo "🔍 This script addresses the 'double free detected in tcache' error"
echo "   that occurs when PyTorch CUDA is used in Ray actors in WSL/Docker."
echo ""

# Step 1: Stop services
echo "1️⃣ Stopping all services..."
docker compose down
echo "   ✅ Services stopped"

# Step 2: Clean up old containers and images
echo ""
echo "2️⃣ Cleaning up old containers..."
docker system prune -f
echo "   ✅ Cleanup completed"

# Step 3: Build stable Ray container
echo ""
echo "3️⃣ Building stable Ray container with CUDA fixes..."
if [ -f "Dockerfile.ray.stable" ]; then
    # Use stable dockerfile if available
    docker build -f Dockerfile.ray.stable -t transcription-ray:stable .
    
    # Update docker-compose to use stable image
    cp docker-compose.yml docker-compose.yml.backup
    sed -i 's|build:.*|image: transcription-ray:stable|g' docker-compose.yml
    
    echo "   ✅ Built stable Ray container"
else
    # Fallback to regular build with fixes
    docker compose build --no-cache ray-head
    echo "   ✅ Built Ray container with fixes"
fi

# Step 4: Start core services
echo ""
echo "4️⃣ Starting core services..."
docker compose up -d mongodb redis
sleep 5

echo "   Starting Ray head node..."
docker compose up -d ray-head
echo "   ✅ Core services started"

# Step 5: Wait for Ray cluster
echo ""
echo "5️⃣ Waiting for Ray cluster to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8265/api/cluster_status > /dev/null 2>&1; then
        echo "   ✅ Ray cluster is ready"
        break
    fi
    
    if [ $i -eq 60 ]; then
        echo "   ❌ Ray cluster timeout"
        echo ""
        echo "🔍 Checking Ray logs..."
        docker compose logs ray-head | tail -20
        exit 1
    fi
    
    echo "   ⏳ Waiting... ($i/60)"
    sleep 3
done

# Step 6: Test CUDA setup
echo ""
echo "6️⃣ Testing CUDA setup in container..."
echo "   Testing nvidia-smi..."
if docker compose exec ray-head nvidia-smi > /dev/null 2>&1; then
    echo "   ✅ nvidia-smi working"
else
    echo "   ⚠️ nvidia-smi not available (CPU mode)"
fi

echo "   Testing PyTorch CUDA..."
docker compose exec ray-head python -c "
import os
import torch
import gc

# Set safe environment
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

print(f'PyTorch version: {torch.__version__}')
device_count = torch.cuda.device_count()
print(f'CUDA device count: {device_count}')

if device_count > 0:
    try:
        # Very safe CUDA test
        x = torch.tensor([1.0], device='cuda:0')
        y = x.cpu()
        del x, y
        torch.cuda.empty_cache()
        gc.collect()
        print('✅ CUDA test passed')
    except Exception as e:
        print(f'⚠️ CUDA test failed: {e}')
        print('Will use CPU mode')
else:
    print('No CUDA devices available')
"

# Step 7: Test Ray actors with safe script
echo ""
echo "7️⃣ Testing Ray actors with safe CUDA handling..."
docker compose exec ray-head python /app/scripts/debug_ray_actors.py

test_result=$?

if [ $test_result -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! CUDA double free issue has been resolved."
    echo ""
    echo "🚀 Starting remaining services..."
    docker compose up -d
    
    echo ""
    echo "✅ All services are now running with CUDA fixes applied."
    echo ""
    echo "🧪 You can now test with:"
    echo "   make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"
    echo ""
    echo "📊 Monitor Ray dashboard at: http://localhost:8265"
    
else
    echo ""
    echo "❌ Tests still failing. Additional debugging needed."
    echo ""
    echo "🔍 Check logs with:"
    echo "   docker compose logs ray-head"
    echo ""
    echo "💡 Try these additional fixes:"
    echo "   1. Ensure Docker has enough memory (>= 8GB)"
    echo "   2. Update NVIDIA drivers"
    echo "   3. Try CPU-only mode: export CUDA_VISIBLE_DEVICES=''"
    
    exit 1
fi

echo ""
echo "🛡️ CUDA Double Free Fix Applied Successfully!"
echo ""
echo "Key fixes implemented:"
echo "   ✅ Stable PyTorch version (2.1.2)"
echo "   ✅ CUDA memory management settings"
echo "   ✅ Safe CUDA detection without torch.cuda.is_available()"
echo "   ✅ Proper environment variables for WSL"
echo "   ✅ Memory cleanup and garbage collection"
echo "   ✅ Conservative CUDA allocation settings"
echo ""