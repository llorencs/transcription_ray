#!/bin/bash

set -e

echo "🚀 Quick Fix for Ray ML Dependencies"
echo "=" * 50

# Check if we have the right CUDA version
echo "1. Checking CUDA compatibility..."
if docker run --rm nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 nvcc --version 2>/dev/null; then
    echo "   ✅ CUDA 12.8.1 base image available"
else
    echo "   ❌ CUDA 12.8.1 base image not available"
    echo "   Make sure you have nvidia-docker properly installed"
fi

# Stop current services
echo "2. Stopping services..."
docker compose down

# Build only Ray container first
echo "3. Building Ray container (this is the critical one)..."
docker compose build --no-cache ray-head

# Test the Ray container specifically
echo "4. Testing Ray container ML dependencies..."
echo "   Testing basic Python..."
docker compose run --rm ray-head python --version

echo "   Testing PyTorch installation..."
docker compose run --rm ray-head python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

echo "   Testing Faster Whisper..."
docker compose run --rm ray-head python -c "
from faster_whisper import WhisperModel
print('Faster Whisper import successful')
"

echo "   Testing Ray..."
docker compose run --rm ray-head python -c "
import ray
print(f'Ray version: {ray.__version__}')
"

echo "   Testing Librosa..."
docker compose run --rm ray-head python -c "
import librosa
print(f'Librosa version: {librosa.__version__}')
"

echo ""
echo "5. If all tests passed, building remaining containers..."
docker compose build --no-cache

echo ""
echo "6. Starting services..."
docker compose up -d

echo ""
echo "✅ Quick fix completed!"
echo ""
echo "🧪 Test with:"
echo "   make debug-actors"
echo "   make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"