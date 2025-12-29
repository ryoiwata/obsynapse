#!/bin/bash
# Start Qdrant vector database in Docker container

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDRANT_STORAGE="${PROJECT_ROOT}/qdrant_storage"

# Create storage directory if it doesn't exist
mkdir -p "${QDRANT_STORAGE}"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^qdrant$"; then
    echo "Qdrant container already exists. Checking if it's running..."
    if docker ps --format '{{.Names}}' | grep -q "^qdrant$"; then
        echo "Qdrant is already running!"
        exit 0
    else
        echo "Starting existing Qdrant container..."
        docker start qdrant
        exit 0
    fi
fi

# Run Qdrant container
echo "Starting Qdrant vector database..."
echo "Storage location: ${QDRANT_STORAGE}"
echo "Access Qdrant UI at: http://localhost:6333/dashboard"

docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "${QDRANT_STORAGE}:/qdrant/storage" \
    qdrant/qdrant

if [ $? -eq 0 ]; then
    echo "✅ Qdrant started successfully!"
    echo "   Container name: qdrant"
    echo "   API endpoint: http://localhost:6333"
    echo "   Dashboard: http://localhost:6333/dashboard"
else
    echo "❌ Failed to start Qdrant container"
    exit 1
fi

