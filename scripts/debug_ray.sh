#!/bin/bash

echo "🔍 Ray Container Debug Script"
echo "=" * 60

echo "1. Checking Ray container status..."
docker compose ps ray-head

echo ""
echo "2. Checking Ray container Python version..."
docker compose exec ray-head python --version

echo ""
echo "3. Checking installed packages in Ray container..."
docker compose exec ray-head pip list | grep -E "(torch|whisper|librosa|numpy|ray)"

echo ""
echo "4. Testing imports in Ray container..."
docker compose exec ray-head python -c "
try:
    import torch
    print('✅ torch imported successfully:', torch.__version__)
except ImportError as e:
    print('❌ torch import failed:', e)

try:
    import faster_whisper
    print('✅ faster_whisper imported successfully')
except ImportError as e:
    print('❌ faster_whisper import failed:', e)

try:
    import ray
    print('✅ ray imported successfully:', ray.__version__)
except ImportError as e:
    print('❌ ray import failed:', e)
"

echo ""
echo "5. Checking requirements file in Ray container..."
docker compose exec ray-head cat /app/requirements.ray.txt

echo ""
echo "6. Checking Python path in Ray container..."
docker compose exec ray-head python -c "import sys; print('\n'.join(sys.path))"

echo ""
echo "7. Checking if Ray is running in container..."
docker compose exec ray-head python -c "
import ray
try:
    if ray.is_initialized():
        print('✅ Ray is initialized')
        print('Cluster resources:', ray.cluster_resources())
    else:
        print('⚠️ Ray is not initialized')
except Exception as e:
    print('❌ Ray check failed:', e)
"

echo ""
echo "Debug completed!"