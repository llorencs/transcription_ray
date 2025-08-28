#!/bin/bash

# scripts/job_submit.sh
# Job submission script for transcription service
# This enables driver logs visibility

set -e

echo "🚀 Submitting Ray Job for Transcription Service..."

# Configuration
RAY_ADDRESS="ray://ray-head:10001"
JOB_NAME="transcription-service-$(date +%Y%m%d-%H%M%S)"
WORKING_DIR="/app"
SCRIPT_PATH="/app/scripts/ray_job_deployment.py"

echo "📋 Job Configuration:"
echo "  • Ray Address: $RAY_ADDRESS"
echo "  • Job Name: $JOB_NAME"
echo "  • Working Directory: $WORKING_DIR"
echo "  • Script Path: $SCRIPT_PATH"

# Submit the job
ray job submit \
    --address="$RAY_ADDRESS" \
    --job-name="$JOB_NAME" \
    --working-dir="$WORKING_DIR" \
    --runtime-env-json='{"pip": ["faster-whisper==1.1.0", "pyannote.audio==3.3.2"], "working_dir": "/app"}' \
    -- python "$SCRIPT_PATH"

echo "✅ Job submitted successfully!"
echo "📊 Monitor your job with:"
echo "  • ray job logs $JOB_NAME --address=$RAY_ADDRESS --follow"
echo "  • Ray Dashboard: http://localhost:8265"