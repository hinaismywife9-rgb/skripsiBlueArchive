#!/bin/bash

# Sentiment Analysis Docker Setup Script

echo "=========================================="
echo "Sentiment Analysis - Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  Docker Compose not found. Will use 'docker compose' instead."
fi

echo "✅ Docker is installed!"
echo ""

# Menu
echo "What would you like to do?"
echo "1. Build Docker image"
echo "2. Run with Docker Compose"
echo "3. Run single container"
echo "4. Stop running containers"
echo "5. View logs"
echo "6. Clean up (remove containers & images)"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "Building Docker image..."
        docker build -t sentiment-analysis:latest .
        echo "✅ Docker image built successfully!"
        ;;
    
    2)
        echo "Starting with Docker Compose..."
        docker-compose up --build
        ;;
    
    3)
        echo "Starting single container..."
        docker run -it --rm \
            -p 8501:8501 \
            -v $(pwd)/sentiment_models:/app/sentiment_models \
            -v $(pwd)/results:/app/results \
            -v $(pwd)/logs:/app/logs \
            sentiment-analysis:latest
        ;;
    
    4)
        echo "Stopping containers..."
        docker-compose down
        echo "✅ Containers stopped!"
        ;;
    
    5)
        echo "Showing logs..."
        docker-compose logs -f
        ;;
    
    6)
        echo "Cleaning up..."
        docker-compose down --volumes
        docker rmi sentiment-analysis:latest
        echo "✅ Cleanup complete!"
        ;;
    
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac
