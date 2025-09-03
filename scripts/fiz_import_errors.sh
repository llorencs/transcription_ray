#!/bin/bash

set -e

echo "🔧 Fixing Import Errors for SafeWhisperActor"
echo "=" * 50

# Step 1: Ensure the safe_whisper_actor.py is in the right places
echo "1️⃣ Copying SafeWhisperActor to correct locations..."

# Create the file in scripts directory
echo "   Creating scripts/safe_whisper_actor.py..."
cat > scripts/safe_whisper_actor.py << 'EOF'
#!/usr/bin/env python3
"""
Safe Whisper Actor that avoids CUDA double free issues.
"""

import ray
import os
import sys
import torch
import gc
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# CRITICAL: Set environment before any CUDA operations
os.environ.update({
    'CUDA_LAUNCH_BLOCKING': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMBA_DISABLE_CUDA': '0'
})

@ray.remote(num_cpus=1, memory=2048*1024*1024)  # 2GB memory limit
class SafeWhisperActor:
    """Safe Whisper actor that avoids CUDA double free errors."""
    
    def __init__(self):
        """Initialize the actor with safe defaults."""
        
        # Set environment in actor process
        os.environ.update({
            'CUDA_LAUNCH_BLOCKING': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1'
        })
        
        self.whisper_model = None
        self.device = "cpu"  # Always use CPU for Whisper
        self.model_size = None
        self.models_dir = Path("/app/models")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"SafeWhisperActor initialized (PID: {os.getpid()})")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information safely."""
        
        info = {
            'pid': os.getpid(),
            'device': self.device,
            'model_loaded': self.whisper_model is not None,
            'model_size': self.model_size
        }
        
        try:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_device_count'] = torch.cuda.device_count()
            
            # Memory info
            import psutil
            process = psutil.Process()
            info['memory_mb'] = process.memory_info().rss / 1024 / 1024
            
        except Exception as e:
            info['error'] = str(e)
            
        return info
    
    def load_model(self, model_size: str = "base", force_cpu: bool = True) -> Dict[str, Any]:
        """Load Whisper model safely."""
        
        try:
            print(f"Loading Whisper model: {model_size}")
            
            from faster_whisper import WhisperModel
            
            # Always use CPU to avoid CUDA issues
            device = "cpu" if force_cpu else self.device
            
            # Load model with conservative settings
            model = WhisperModel(
                model_size,
                device=device,
                download_root=str(self.models_dir),
                compute_type="int8" if device == "cpu" else "float16"
            )
            
            # Clean up any existing model
            if self.whisper_model is not None:
                del self.whisper_model
                gc.collect()
            
            self.whisper_model = model
            self.model_size = model_size
            self.device = device
            
            print(f"✅ Model loaded: {model_size} on {device}")
            
            return {
                'success': True,
                'model_size': model_size,
                'device': device,
                'compute_type': "int8" if device == "cpu" else "float16"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Model loading failed: {error_msg}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'model_size': model_size
            }
    
    def transcribe_file(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file safely."""
        
        if self.whisper_model is None:
            return {
                'success': False,
                'error': 'Model not loaded. Call load_model() first.'
            }
        
        try:
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                return {
                    'success': False,
                    'error': f'Audio file not found: {audio_path}'
                }
            
            print(f"Transcribing: {audio_path}")
            
            # Transcription parameters with safe defaults
            transcribe_params = {
                'beam_size': 5,
                'language': None,
                'task': 'transcribe',
                'vad_filter': True,
                'vad_parameters': dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                )
            }
            
            # Update with provided kwargs
            transcribe_params.update(kwargs)
            
            # Perform transcription
            segments, info = self.whisper_model.transcribe(
                str(audio_path),
                **transcribe_params
            )
            
            # Collect results
            transcript_segments = []
            full_text = []
            
            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'avg_logprob': segment.avg_logprob,
                    'no_speech_prob': segment.no_speech_prob
                }
                transcript_segments.append(segment_data)
                full_text.append(segment.text.strip())
            
            result = {
                'success': True,
                'file': str(audio_path),
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'full_text': ' '.join(full_text),
                'segments': transcript_segments,
                'model_size': self.model_size,
                'device': self.device
            }
            
            print(f"✅ Transcription completed: {len(transcript_segments)} segments")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Transcription failed: {error_msg}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'file': str(audio_path) if 'audio_path' in locals() else 'unknown'
            }
    
    def cleanup(self) -> Dict[str, Any]:
        """Clean up resources."""
        
        try:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            gc.collect()
            
            try:
                import torch
                if torch.cuda.device_count() > 0:
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore CUDA cleanup errors
            
            print("✅ Cleanup completed")
            
            return {
                'success': True,
                'message': 'Resources cleaned up'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
EOF

echo "   ✅ Created scripts/safe_whisper_actor.py"

# Also create it in src/services for easier import
echo "   Creating src/services/safe_whisper_actor.py..."
cp scripts/safe_whisper_actor.py src/services/safe_whisper_actor.py
echo "   ✅ Created src/services/safe_whisper_actor.py"

# Step 2: Fix the transcription_service.py import
echo ""
echo "2️⃣ Fixing transcription_service.py imports..."

cat > src/services/transcription_service_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
Fixed Transcription Service using SafeWhisperActor.
"""

import ray
import os
import sys
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
sys.path.append('/app/src')

# Import the safe actor - multiple fallback paths
try:
    from .safe_whisper_actor import SafeWhisperActor
    print("✅ Imported SafeWhisperActor from services")
except ImportError:
    try:
        from src.services.safe_whisper_actor import SafeWhisperActor
        print("✅ Imported SafeWhisperActor from src.services")
    except ImportError:
        try:
            sys.path.append('/app/scripts')
            from safe_whisper_actor import SafeWhisperActor
            print("✅ Imported SafeWhisperActor from scripts")
        except ImportError as e:
            print(f"❌ Failed to import SafeWhisperActor: {e}")
            raise ImportError(f"Cannot import SafeWhisperActor. Make sure the file exists in scripts/ or src/services/")

class TranscriptionService:
    """Transcription service using SafeWhisperActor to avoid CUDA issues."""
    
    def __init__(self, ray_address: Optional[str] = None):
        """Initialize the service."""
        
        self.ray_address = ray_address or "ray://ray-head:10001"
        self.actors: Dict[str, Any] = {}
        self.actor_pool_size = 2
        self.default_model_size = "base"
        
        print(f"TranscriptionService initializing (Ray: {self.ray_address})")
    
    async def initialize(self) -> bool:
        """Initialize Ray connection and actor pool."""
        
        try:
            # Connect to Ray cluster
            if not ray.is_initialized():
                print("Connecting to Ray cluster...")
                ray.init(address=self.ray_address, ignore_reinit_error=True)
                print("✅ Connected to Ray cluster")
            
            # Create actor pool
            print(f"Creating {self.actor_pool_size} SafeWhisperActors...")
            for i in range(self.actor_pool_size):
                actor_id = f"whisper_actor_{i}"
                actor = SafeWhisperActor.remote()
                self.actors[actor_id] = actor
                print(f"   Created {actor_id}")
            
            # Load models in all actors
            print("Loading Whisper models...")
            load_tasks = []
            for actor_id, actor in self.actors.items():
                task = actor.load_model.remote(self.default_model_size, force_cpu=True)
                load_tasks.append((actor_id, task))
            
            # Wait for all models to load
            all_loaded = True
            for actor_id, task in load_tasks:
                try:
                    result = ray.get(task, timeout=120)
                    if result['success']:
                        print(f"   ✅ {actor_id}: Model loaded")
                    else:
                        print(f"   ❌ {actor_id}: Load failed - {result.get('error')}")
                        all_loaded = False
                except Exception as e:
                    print(f"   ❌ {actor_id}: Exception - {e}")
                    all_loaded = False
            
            if not all_loaded:
                print("⚠️ Some actors failed to load models, but continuing...")
            
            print("✅ TranscriptionService initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize TranscriptionService: {e}")
            traceback.print_exc()
            return False
    
    def get_available_actor(self) -> Optional[Any]:
        """Get an available actor from the pool."""
        
        if not self.actors:
            return None
        
        actor_ids = list(self.actors.keys())
        selected_id = actor_ids[0]
        return self.actors[selected_id]
    
    async def transcribe_file(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        task: str = 'transcribe',
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe an audio file."""
        
        try:
            actor = self.get_available_actor()
            if not actor:
                return {
                    'success': False,
                    'error': 'No available actors',
                    'timestamp': datetime.now().isoformat()
                }
            
            transcribe_params = {
                'language': language,
                'task': task,
                'beam_size': kwargs.get('beam_size', 5),
                'vad_filter': kwargs.get('vad_filter', True)
            }
            
            print(f"Transcribing {audio_path} with params: {transcribe_params}")
            
            task_future = actor.transcribe_file.remote(audio_path, **transcribe_params)
            result = ray.get(task_future, timeout=300)
            
            result['timestamp'] = datetime.now().isoformat()
            result['service'] = 'TranscriptionService'
            result['parameters'] = transcribe_params
            
            if result['success']:
                print(f"✅ Transcription completed: {len(result.get('segments', []))} segments")
            else:
                print(f"❌ Transcription failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Transcription service error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'file': audio_path,
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status and actor health."""
        
        status = {
            'service': 'TranscriptionService',
            'ray_initialized': ray.is_initialized(),
            'ray_address': self.ray_address,
            'actor_count': len(self.actors),
            'timestamp': datetime.now().isoformat()
        }
        
        actor_status = {}
        for actor_id, actor in self.actors.items():
            try:
                info_task = actor.get_system_info.remote()
                info = ray.get(info_task, timeout=10)
                actor_status[actor_id] = {
                    'healthy': True,
                    'info': info
                }
            except Exception as e:
                actor_status[actor_id] = {
                    'healthy': False,
                    'error': str(e)
                }
        
        status['actors'] = actor_status
        
        if ray.is_initialized():
            try:
                status['cluster_resources'] = ray.cluster_resources()
                status['available_resources'] = ray.available_resources()
            except Exception as e:
                status['cluster_error'] = str(e)
        
        return status
    
    async def cleanup(self) -> bool:
        """Clean up service resources."""
        
        try:
            print("Cleaning up TranscriptionService...")
            
            cleanup_tasks = []
            for actor_id, actor in self.actors.items():
                task = actor.cleanup.remote()
                cleanup_tasks.append((actor_id, task))
            
            for actor_id, task in cleanup_tasks:
                try:
                    ray.get(task, timeout=30)
                    print(f"   ✅ {actor_id}: Cleaned up")
                except Exception as e:
                    print(f"   ❌ {actor_id}: Cleanup failed - {e}")
            
            self.actors.clear()
            print("✅ TranscriptionService cleanup completed")
            return True
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False

# Alias for compatibility
RayTranscriptionService = TranscriptionService
EOF

echo "   ✅ Created src/services/transcription_service_fixed.py"

# Step 3: Replace the original file
echo ""
echo "3️⃣ Replacing original transcription_service.py..."
mv src/services/transcription_service.py src/services/transcription_service_backup.py
mv src/services/transcription_service_fixed.py src/services/transcription_service.py
echo "   ✅ Replaced transcription_service.py"

# Step 4: Test the imports
echo ""
echo "4️⃣ Testing imports in container..."
docker compose exec api python -c "
import sys
sys.path.append('/app/src')

print('Testing SafeWhisperActor import...')
try:
    from src.services.safe_whisper_actor import SafeWhisperActor
    print('✅ SafeWhisperActor import successful')
except ImportError as e:
    print(f'❌ SafeWhisperActor import failed: {e}')
    exit(1)

print('Testing TranscriptionService import...')
try:
    from src.services.transcription_service import TranscriptionService
    print('✅ TranscriptionService import successful')
except ImportError as e:
    print(f'❌ TranscriptionService import failed: {e}')
    exit(1)

print('All imports successful!')
"

if [ $? -eq 0 ]; then
    echo "   ✅ Import tests passed"
else
    echo "   ❌ Import tests failed - checking file structure..."
    
    echo ""
    echo "📁 Current file structure:"
    echo "scripts/"
    ls -la scripts/safe_whisper_actor.py 2>/dev/null || echo "   ❌ scripts/safe_whisper_actor.py missing"
    
    echo "src/services/"
    ls -la src/services/safe_whisper_actor.py 2>/dev/null || echo "   ❌ src/services/safe_whisper_actor.py missing"
    ls -la src/services/transcription_service.py 2>/dev/null || echo "   ❌ src/services/transcription_service.py missing"
    
    exit 1
fi

# Step 5: Restart API container
echo ""
echo "5️⃣ Restarting API container..."
docker compose restart api
echo "   ✅ API container restarted"

# Wait for container to be ready
echo "   Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ⚠️ API timeout - check logs with: docker compose logs api"
    fi
    sleep 2
done

echo ""
echo "🎉 Import Error Fix Complete!"
echo ""
echo "✅ Fixed issues:"
echo "   ✅ Created SafeWhisperActor in multiple locations"
echo "   ✅ Fixed import paths in TranscriptionService"
echo "   ✅ Added fallback import mechanisms"
echo "   ✅ Tested imports successfully"
echo "   ✅ Restarted API service"
echo ""
echo "🧪 Test the API now:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/api/v1/transcription/status"
echo ""