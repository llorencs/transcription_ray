#!/bin/bash

echo "🔍 Testing Ray Actors Step by Step"
echo "=" * 60

# Step 1: Test basic actor functionality
echo "1. Testing simple actor inside Ray container..."
docker compose exec ray-head python /app/scripts/test_simple_actor.py

if [ $? -eq 0 ]; then
    echo "   ✅ Simple actor test passed"
else
    echo "   ❌ Simple actor test failed"
    exit 1
fi

# Step 2: Test memory and resource usage
echo ""
echo "2. Checking Ray cluster resources..."
docker compose exec ray-head python -c "
import ray
if not ray.is_initialized():
    ray.init(address='ray://ray-head:10001', ignore_reinit_error=True)

print('Cluster resources:', ray.cluster_resources())
print('Available resources:', ray.available_resources())

# Check memory usage
import psutil
memory = psutil.virtual_memory()
print(f'System memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available')
print(f'Memory usage: {memory.percent}%')
"

# Step 3: Test GPU availability in container
echo ""
echo "3. Testing GPU availability..."
docker compose exec ray-head python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}, Memory: {props.total_memory / (1024**3):.1f} GB')
        
    # Test CUDA functionality
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print('✅ CUDA computation test passed')
    except Exception as e:
        print(f'❌ CUDA computation test failed: {e}')
else:
    print('GPU not available - will use CPU mode')
"

# Step 4: Test Whisper model loading specifically
echo ""
echo "4. Testing Whisper model loading in container..."
docker compose exec ray-head python -c "
import os
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

try:
    from faster_whisper import WhisperModel
    print('✅ faster_whisper imported successfully')
    
    # Create models directory
    os.makedirs('/app/models/whisper', exist_ok=True)
    
    print('Loading Whisper base model on CPU...')
    model = WhisperModel('base', device='cpu', download_root='/app/models')
    print('✅ Whisper model loaded successfully on CPU')
    
    # Test if GPU model loading works
    try:
        import torch
        if torch.cuda.is_available():
            print('Testing GPU model loading...')
            gpu_model = WhisperModel('base', device='cuda', download_root='/app/models')
            print('✅ Whisper model loaded successfully on GPU')
        else:
            print('⚠️ GPU not available, skipping GPU model test')
    except Exception as gpu_e:
        print(f'⚠️ GPU model loading failed: {gpu_e}')
        
except Exception as e:
    print(f'❌ Whisper model loading failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "5. Testing debug actors script inside container..."
docker compose exec ray-head python /app/scripts/debug_ray_actors.py

if [ $? -eq 0 ]; then
    echo "   ✅ Debug actors test passed"
else
    echo "   ❌ Debug actors test failed"
fi

echo ""
echo "✅ Step-by-step actor testing completed!"
echo ""
echo "🧪 If all tests passed, try:"
echo "   make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"