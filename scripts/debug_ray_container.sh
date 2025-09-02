#!/bin/bash

echo "🔍 Running Ray actors debug inside Ray container..."
echo "=" * 60

# Execute the debug script inside the Ray container where all dependencies should be available
docker compose exec ray-head python /app/scripts/debug_ray_actors.py

echo ""
echo "Debug completed!"