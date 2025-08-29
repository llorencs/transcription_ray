#!/bin/bash

# scripts/job_management.sh
# Job management script for transcription service

RAY_ADDRESS="ray://ray-head:10001"

function show_help() {
    echo "Ray Job Management for Transcription Service"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  submit    Submit a new transcription service job"
    echo "  list      List all jobs"
    echo "  logs      Show logs for latest job (with follow)"
    echo "  status    Show status of latest job"
    echo "  stop      Stop the latest job"
    echo "  cleanup   Clean up old jobs"
    echo "  help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 submit"
    echo "  $0 logs"
    echo "  $0 status"
}

function submit_job() {
    echo "🚀 Submitting transcription service job..."
    JOB_SUBMISSION_ID="transcription-service-$(date +%Y%m%d-%H%M%S)"
    
    # FIXED: Use --submission-id instead of --job-name
    ray job submit \
        --address="$RAY_ADDRESS" \
        --submission-id="$JOB_SUBMISSION_ID" \
        --working-dir="/app" \
        --runtime-env-json='{"working_dir": "/app", "pip": ["fastapi", "uvicorn"]}' \
        -- python /app/scripts/ray_job_deployment.py
        
    echo "✅ Job '$JOB_SUBMISSION_ID' submitted"
    echo "📊 Monitor with: $0 logs"
}

function list_jobs() {
    echo "📋 Active Ray Jobs:"
    ray job list --address="$RAY_ADDRESS"
}

function show_logs() {
    echo "📝 Getting latest job logs..."
    
    # Get the latest transcription service job
    LATEST_JOB=$(ray job list --address="$RAY_ADDRESS" --format json 2>/dev/null | \
        jq -r '.[] | select(.submission_id | test("transcription-service")) | .submission_id' 2>/dev/null | \
        head -1)
    
    if [ -z "$LATEST_JOB" ]; then
        echo "❌ No transcription service jobs found"
        echo "Available jobs:"
        ray job list --address="$RAY_ADDRESS"
        return 1
    fi
    
    echo "📄 Following logs for job: $LATEST_JOB"
    ray job logs "$LATEST_JOB" --address="$RAY_ADDRESS" --follow
}

function show_status() {
    echo "📊 Job Status:"
    
    LATEST_JOB=$(ray job list --address="$RAY_ADDRESS" --format json 2>/dev/null | \
        jq -r '.[] | select(.submission_id | test("transcription-service")) | .submission_id' 2>/dev/null | \
        head -1)
    
    if [ -z "$LATEST_JOB" ]; then
        echo "❌ No transcription service jobs found"
        echo "Available jobs:"
        ray job list --address="$RAY_ADDRESS"
        return 1
    fi
    
    ray job status "$LATEST_JOB" --address="$RAY_ADDRESS"
}

function stop_job() {
    echo "🛑 Stopping latest transcription service job..."
    
    LATEST_JOB=$(ray job list --address="$RAY_ADDRESS" --format json 2>/dev/null | \
        jq -r '.[] | select(.submission_id | test("transcription-service")) | .submission_id' 2>/dev/null | \
        head -1)
    
    if [ -z "$LATEST_JOB" ]; then
        echo "❌ No transcription service jobs found"
        return 1
    fi
    
    ray job stop "$LATEST_JOB" --address="$RAY_ADDRESS"
    echo "✅ Job stopped: $LATEST_JOB"
}

function cleanup_jobs() {
    echo "🧹 Cleaning up old jobs..."
    
    # List all transcription service jobs
    OLD_JOBS=$(ray job list --address="$RAY_ADDRESS" --format json 2>/dev/null | \
        jq -r '.[] | select(.submission_id | test("transcription-service")) | select(.status == "SUCCEEDED" or .status == "FAILED") | .submission_id' 2>/dev/null)
    
    if [ -z "$OLD_JOBS" ]; then
        echo "ℹ️  No old jobs to clean up"
        return 0
    fi
    
    echo "🗑️  Found old jobs to clean up:"
    echo "$OLD_JOBS"
    
    for job_id in $OLD_JOBS; do
        echo "Deleting job: $job_id"
        ray job delete "$job_id" --address="$RAY_ADDRESS" --yes
    done
    
    echo "✅ Cleanup completed"
}

# Main command handling
case "${1:-help}" in
    submit)
        submit_job
        ;;
    list)
        list_jobs
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    stop)
        stop_job
        ;;
    cleanup)
        cleanup_jobs
        ;;
    help|*)
        show_help
        ;;
esac