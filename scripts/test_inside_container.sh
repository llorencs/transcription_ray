#!/bin/bash

# Test script to run inside the API container

echo "🧪 Testing ML dependencies inside API container..."
echo "=" * 60

# Test Python and basic imports
echo "🐍 Python version:"
python --version

echo ""
echo "📦 Testing ML imports:"

# Test each import individually
python -c "
import sys
print('Testing imports...')

try:
    import torch
    print('✅ PyTorch: OK - version', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ PyTorch: FAILED -', e)

try:
    import faster_whisper
    print('✅ faster_whisper: OK')
except ImportError as e:
    print('❌ faster_whisper: FAILED -', e)

try:
    import librosa
    print('✅ librosa: OK - version', librosa.__version__)
except ImportError as e:
    print('❌ librosa: FAILED -', e)

try:
    import soundfile
    print('✅ soundfile: OK')
except ImportError as e:
    print('❌ soundfile: FAILED -', e)

try:
    import numpy
    print('✅ numpy: OK - version', numpy.__version__)
except ImportError as e:
    print('❌ numpy: FAILED -', e)

print('')
print('🤖 Testing Whisper model loading...')
try:
    from faster_whisper import WhisperModel
    import os
    
    # Create models directory
    os.makedirs('/app/models/whisper', exist_ok=True)
    
    # Try to load base model on CPU
    print('Loading base model on CPU...')
    model = WhisperModel('base', device='cpu', download_root='/app/models')
    print('✅ Whisper model loaded successfully!')
    
except Exception as e:
    print('❌ Whisper model loading failed:', e)
    import traceback
    traceback.print_exc()

print('')
print('📁 Checking directories:')
import os
dirs_to_check = ['/app/temp', '/app/models', '/app/src']
for directory in dirs_to_check:
    if os.path.exists(directory):
        print(f'✅ {directory}: exists')
    else:
        print(f'❌ {directory}: missing')

print('')
print('🔧 Environment variables:')
env_vars = ['PYTHONPATH', 'CUDA_HOME', 'PATH']
for var in env_vars:
    value = os.getenv(var, 'Not set')
    print(f'   {var}: {value}')
"

echo ""
echo "🏥 System information:"
echo "Memory usage:"
free -h || echo "free command not available"

echo ""
echo "Disk usage:"
df -h / || echo "df command not available"

echo ""
echo "GPU information:"
nvidia-smi || echo "nvidia-smi not available (running on CPU)"

echo ""
echo "✅ Container test completed!"