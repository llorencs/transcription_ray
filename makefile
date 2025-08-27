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
	@echo "  clean        - Clean up containers and volumes"
	@echo "  build        - Build Docker images"
	@echo "  deploy       - Deploy Ray Serve models"
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

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	@docker compose down -v --remove-orphans
	@docker system prune -f
	@sudo rm -rf temp/* logs/*

# Build images
build:
	@echo "🏗️ Building Docker images..."
	@docker compose build --parallel

# Deploy models
deploy:
	@echo "🤖 Deploying Ray Serve models..."
	@docker compose exec ray-head python src/deployments/ray_serve_models.py

# Health check
health:
	@echo "🔍 Checking service health..."
	@curl -s http://localhost:8080/health | jq '.' || echo "❌ API service not responding"
	@curl -s http://localhost:8265/api/cluster_status | jq '.cluster_status' || echo "❌ Ray cluster not responding"

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

# Backup operations
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker compose exec -T mongodb mongodump --db transcription_db --archive | gzip > backups/transcription_$(shell date +%Y%m%d_%H%M%S).gz

restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please specify BACKUP_FILE"; \
		echo "Example: make restore BACKUP_FILE=backups/transcription_20240101_120000.gz"; \
		exit 1; \
	fi
	@echo "📥 Restoring from backup: $(BACKUP_FILE)"
	@gunzip -c "$(BACKUP_FILE)" | docker compose exec -T mongodb mongorestore --archive --drop

# Monitoring
monitor:
	@echo "📊 Opening monitoring dashboards..."
	@echo "Ray Dashboard: http://localhost:8265"
	@echo "API Documentation: http://localhost:8080/docs"
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:8080/docs; \
		open http://localhost:8265; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:8080/docs; \
		xdg-open http://localhost:8265; \
	fi

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
	@echo "     \"diarize\": true,"
	@echo "     \"preprocess\": true"
	@echo "   }'"
	@echo ""
	@echo "3. Check status:"
	@echo "   curl http://localhost:8080/tasks/your-task-id"
	@echo ""
	@echo "4. Download results:"
	@echo "   curl http://localhost:8080/results/your-task-id/srt"