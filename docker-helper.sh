#!/bin/bash

# Docker Helper Script untuk Sentiment Analysis Dashboard
# Linux/Mac bash script

set -e

show_help() {
    cat << EOF

=====================================
  SENTIMENT ANALYSIS - DOCKER HELPER
=====================================

Usage: ./docker-helper.sh [command]

Commands:
  build       - Build Docker image
  up          - Start containers (docker-compose up -d)
  down        - Stop containers (docker-compose down)
  restart     - Restart containers
  logs        - View container logs (follow)
  shell       - Open bash shell in container
  status      - Check container status
  verify      - Verify setup and models
  clean       - Remove containers and image
  push        - Push image to Docker Hub
  pull        - Pull image from Docker Hub
  help        - Show this help message

Examples:
  ./docker-helper.sh build      # Build image first time
  ./docker-helper.sh up         # Start dashboard
  ./docker-helper.sh logs       # View logs
  ./docker-helper.sh down       # Stop dashboard

EOF
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    build)
        echo "🔨 Building Docker image..."
        docker-compose build
        echo "✓ Build complete!"
        ;;
    
    up)
        echo "🚀 Starting containers..."
        docker-compose up -d
        echo ""
        echo "✓ Containers started!"
        echo "📊 Dashboard: http://localhost:8501"
        echo "⏳ Waiting 40 seconds for models to load..."
        sleep 40
        echo ""
        docker-compose ps
        ;;
    
    down)
        echo "⛔ Stopping containers..."
        docker-compose down
        echo "✓ Containers stopped!"
        ;;
    
    restart)
        echo "🔄 Restarting containers..."
        docker-compose restart
        echo "✓ Containers restarted!"
        echo "⏳ Waiting for models to load..."
        sleep 40
        ;;
    
    logs)
        echo "📋 Showing container logs (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    
    shell)
        echo "🐚 Opening bash shell in container..."
        docker-compose exec sentiment-dashboard bash
        ;;
    
    status)
        echo ""
        echo "Container Status:"
        echo "================="
        docker-compose ps
        echo ""
        echo "Latest Logs (last 20 lines):"
        echo "============================"
        docker-compose logs --tail 20
        ;;
    
    verify)
        echo "🔍 Verifying setup in container..."
        docker-compose exec sentiment-dashboard python verify_setup.py
        ;;
    
    clean)
        echo ""
        echo "⚠️  WARNING: This will remove containers and images"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down
            docker system prune -f
            echo "✓ Cleanup complete!"
        fi
        ;;
    
    push)
        if [ -z "$2" ]; then
            echo "Usage: ./docker-helper.sh push <registry/username>"
            echo "Example: ./docker-helper.sh push myregistry/sentiment-analysis"
            exit 1
        fi
        echo "📤 Pushing image to registry..."
        docker tag sentiment-analysis:latest "$2:latest"
        docker push "$2:latest"
        echo "✓ Push complete!"
        ;;
    
    pull)
        if [ -z "$2" ]; then
            echo "Usage: ./docker-helper.sh pull <registry/username>"
            echo "Example: ./docker-helper.sh pull myregistry/sentiment-analysis"
            exit 1
        fi
        echo "📥 Pulling image from registry..."
        docker pull "$2:latest"
        echo "✓ Pull complete!"
        ;;
    
    help)
        show_help
        ;;
    
    *)
        echo "❌ Unknown command: $1"
        echo "Run './docker-helper.sh help' for usage"
        exit 1
        ;;
esac
