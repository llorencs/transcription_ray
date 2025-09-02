#!/bin/bash

echo "🔍 Verifying build and dependencies..."
echo "=" * 50

# Check if Ray container built successfully
echo "1. Checking Ray container build..."
if docker compose images ray-head | grep -q ray-head; then
    echo "   ✅ Ray container built"
else
    echo "   ❌ Ray container not built"
    exit 1
fi

# Test basic container startup
echo "2. Testing container startup..."
if docker compose run --rm ray-head python --version; then
    echo "   ✅ Python available"
else
    echo "   ❌ Python not available"
    exit 1
fi

# Test ML imports
echo "3. Testing ML dependencies..."
docker compose run --rm ray-head python -c "
try:
    import torch
    print('   ✅ torch:', torch.__version__)
    print('   ✅ CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('   ❌ torch failed:', e)
    exit(1)

try:
    import faster_whisper
    print('   ✅ faster_whisper: OK')
except ImportError as e:
    print('   ❌ faster_whisper failed:', e)
    exit(1)

try:
    import librosa
    print('   ✅ librosa:', librosa.__version__)
except ImportError as e:
    print('   ❌ librosa failed:', e)
    exit(1)

try:
    import ray
    print('   ✅ ray:', ray.__version__)
except ImportError as e:
    print('   ❌ ray failed:', e)
    exit(1)

print('✅ All ML dependencies available')
"

if [ $? -eq 0 ]; then
    echo "   ✅ All ML dependencies verified"
else
    echo "   ❌ ML dependencies verification failed"
    exit 1
fi

echo ""
echo "✅ Build verification completed successfully!"