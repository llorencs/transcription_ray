#!/bin/bash

# scripts/job_submit.sh
# Job submission script for transcription service
# This enables driver logs visibility

set -e

echo "🚀 Submitting Ray Job for Transcription Service..."

# Configuration
RAY_ADDRESS="ray://ray-head:10001"
JOB_SUBMISSION_ID="transcription-service-$(date +%Y%m%d-%H%M%S)"
WORKING_DIR="/app"
SCRIPT_PATH="/app/scripts/ray_job_deployment.py"

echo "📋 Job Configuration:"
echo "  • Ray Address: $RAY_ADDRESS"
echo "  • Job Submission ID: $JOB_SUBMISSION_ID"
echo "  • Working Directory: $WORKING_DIR"
echo "  • Script Path: $SCRIPT_PATH"

# Submit the job - FIXED: Use --submission-id instead of --job-name
ray job submit \
    --address="$RAY_ADDRESS" \
    --submission-id="$JOB_SUBMISSION_ID" \
    --working-dir="$WORKING_DIR" \
    --runtime-env-json='{"pip": ["faster-whisper==1.1.0", "pyannote.audio==3.3.2"], "working_dir": "/app"}' \
    -- python "$SCRIPT_PATH"

echo "✅ Job submitted successfully!"
echo "📊 Monitor your job with:"
echo "  • ray job logs $JOB_SUBMISSION_ID --address=$RAY_ADDRESS --follow"
echo "  • Ray Dashboard: http://localhost:8265"