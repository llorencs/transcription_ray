.PHONY: help start stop restart logs status test clean build

# Default target
help:
	@echo "🚀 Advanced Transcription Service Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  start        - Start all services"
	@echo "  stop         - Stop all services"
	@echo "  restart      - Restart all services"
	@echo "  logs         - Show service logs"
	@echo "  status       - Show service status"
	@echo "  test         - Run API tests (requires TEST_AUDIO_FILE)"
	@echo "  test-ray     - Run Ray-based transcription test"
	@echo "  debug-ray    - Debug Ray container and dependencies"
	@echo "  debug-actors - Debug Ray actors specifically"
	@echo "  verify-ray   - Verify ML dependencies in Ray container"
	@echo "  clean        - Clean up containers and volumes"
	@echo "  build        - Build Docker images"
	@echo "  health       - Check service health"

# Start services
start:
	@echo "🚀 Starting Advanced Transcription Service..."
	@chmod +x scripts/start.sh
	@./scripts/start.sh

# Stop services
stop:
	@echo "🛑 Stopping services..."
	@docker compose down

# Restart services
restart: stop start

# Show logs
logs:
	@docker compose logs -f

# Show service status
status:
	@docker compose ps

# Run tests
test:
	@if [ -z "$(TEST_AUDIO_FILE)" ]; then \
		echo "❌ Please set TEST_AUDIO_FILE environment variable"; \
		echo "Example: make test TEST_AUDIO_FILE=/path/to/audio.wav"; \
		exit 1; \
	fi
	@echo "🧪 Running API tests..."
	@python3 scripts/test_api.py --audio-file "$(TEST_AUDIO_FILE)"

# Run Ray transcription test
test-ray:
	@if [ -z "$(TEST_AUDIO_FILE)" ]; then \
		echo "❌ Please set TEST_AUDIO_FILE environment variable"; \
		echo "Example: make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"; \
		exit 1; \
	fi
	@echo "🧪 Running Ray-based transcription test..."
	@python3 scripts/test_ray.py "$(TEST_AUDIO_FILE)"

# Debug Ray container
debug-ray:
	@echo "🔍 Debugging Ray container..."
	@bash scripts/debug_ray.sh

# Simple Ray test
test-ray-simple:
	@echo "🧪 Running simple Ray test..."
	@python3 scripts/test_ray_simple.py

# Debug Ray actors specifically
debug-actors:
	@echo "🔍 Debugging Ray actors..."
	@python3 scripts/debug_ray_actors.py

# Verify Ray dependencies
verify-ray:
	@echo "🔍 Verifying ML dependencies in Ray container..."
	@docker compose exec ray-head python /app/scripts/verify_ray_deps.py

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	@docker compose down -v --remove-orphans
	@docker system prune -f
	@sudo rm -rf temp/* logs/* test_result_*.srt test_result_*.txt

# Build images
build:
	@echo "🏗️ Building Docker images..."
	@docker compose build --parallel

# Health check
health:
	@echo "🔍 Checking service health..."
	@curl -s http://localhost:8080/health | jq '.' || echo "❌ API service not responding"
	@curl -s http://localhost:8000/health | jq '.' || echo "❌ Ray Serve not responding (optional)"
	@curl -s http://localhost:8265/api/cluster_status | jq '.cluster_status' || echo "❌ Ray cluster not responding (optional)"

# Development targets
dev-start:
	@echo "🔧 Starting development environment..."
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-logs:
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Database operations
db-reset:
	@echo "🗄️ Resetting database..."
	@docker compose exec mongodb mongosh transcription_db --eval "db.dropDatabase()"
	@docker compose exec mongodb mongosh transcription_db < mongo-init.js

# Example usage
example:
	@echo "📝 Example API usage:"
	@echo ""
	@echo "1. Upload file:"
	@echo "   curl -X POST http://localhost:8080/files/upload -F 'file=@audio.wav'"
	@echo ""
	@echo "2. Start transcription:"
	@echo "   curl -X POST http://localhost:8080/transcribe -H 'Content-Type: application/json' -d '{"
	@echo "     \"file_id\": \"your-file-id\","
	@echo "     \"model\": \"base\","
	@echo "     \"diarize\": false,"
	@echo "     \"preprocess\": false"
	@echo "   }'"
	@echo ""
	@echo "3. Check status:"
	@echo "   curl http://localhost:8080/tasks/your-task-id"
	@echo ""
	@echo "4. Download results:"
	@echo "   curl http://localhost:8080/results/your-task-id/srt"
	@echo ""
	@echo "Test commands:"
	@echo "  make test-ray TEST_AUDIO_FILE=/path/to/audio.wav"