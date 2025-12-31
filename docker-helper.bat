@echo off
REM Docker Helper Script untuk Sentiment Analysis Dashboard
REM Windows batch file untuk memudahkan Docker operations

setlocal enabledelayedexpansion

if "%1"=="" (
    echo.
    echo =====================================
    echo   SENTIMENT ANALYSIS - DOCKER HELPER
    echo =====================================
    echo.
    echo Usage: docker-helper.bat [command]
    echo.
    echo Commands:
    echo   build       - Build Docker image
    echo   up          - Start containers (docker-compose up -d)
    echo   down        - Stop containers (docker-compose down)
    echo   restart     - Restart containers
    echo   logs        - View container logs
    echo   shell       - Open bash shell in container
    echo   status      - Check container status
    echo   verify      - Verify setup and models
    echo   clean       - Remove containers and image
    echo   help        - Show this help message
    echo.
    goto :eof
)

if "%1"=="build" (
    echo Building Docker image...
    docker-compose build
    echo ✓ Build complete!
    goto :eof
)

if "%1"=="up" (
    echo Starting containers...
    docker-compose up -d
    echo.
    echo ✓ Containers started!
    echo Dashboard will be available at: http://localhost:8501
    echo Waiting 40 seconds for models to load...
    timeout /t 40 /nobreak
    echo.
    docker-compose ps
    goto :eof
)

if "%1"=="down" (
    echo Stopping containers...
    docker-compose down
    echo ✓ Containers stopped!
    goto :eof
)

if "%1"=="restart" (
    echo Restarting containers...
    docker-compose restart
    echo ✓ Containers restarted!
    echo Waiting for models to load...
    timeout /t 40 /nobreak
    goto :eof
)

if "%1"=="logs" (
    echo Showing container logs (Ctrl+C to exit)...
    docker-compose logs -f
    goto :eof
)

if "%1"=="shell" (
    echo Opening bash shell in container...
    docker-compose exec sentiment-dashboard bash
    goto :eof
)

if "%1"=="status" (
    echo.
    echo Container Status:
    echo ================
    docker-compose ps
    echo.
    echo Latest Logs:
    echo ============
    docker-compose logs --tail 20
    goto :eof
)

if "%1"=="verify" (
    echo Verifying setup in container...
    docker-compose exec sentiment-dashboard python verify_setup.py
    goto :eof
)

if "%1"=="clean" (
    echo.
    echo WARNING: This will remove containers and images
    echo.
    docker-compose down
    docker system prune -f
    echo ✓ Cleanup complete!
    goto :eof
)

if "%1"=="help" (
    goto :help
)

echo Unknown command: %1
echo Run "docker-helper.bat help" for usage
