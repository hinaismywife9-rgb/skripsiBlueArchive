@echo off
REM Sentiment Analysis - Docker Setup Script for Windows

echo ==========================================
echo Sentiment Analysis - Docker Setup (Windows)
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker is not installed or not in PATH!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker is installed!
echo.

REM Menu
echo What would you like to do?
echo 1. Build Docker image
echo 2. Run with Docker Compose
echo 3. Run single container
echo 4. Stop running containers
echo 5. View logs
echo 6. Clean up (remove containers and images)
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo Building Docker image...
    docker build -t sentiment-analysis:latest .
    echo.
    echo Successfully built! Run with: docker run -it -p 8501:8501 sentiment-analysis:latest
    pause
) else if "%choice%"=="2" (
    echo Starting with Docker Compose...
    docker-compose up --build
) else if "%choice%"=="3" (
    echo Starting single container...
    docker run -it --rm ^
        -p 8501:8501 ^
        -v %cd%\sentiment_models:/app/sentiment_models ^
        -v %cd%\results:/app/results ^
        -v %cd%\logs:/app/logs ^
        sentiment-analysis:latest
) else if "%choice%"=="4" (
    echo Stopping containers...
    docker-compose down
    echo.
    echo Containers stopped!
    pause
) else if "%choice%"=="5" (
    echo Showing logs...
    docker-compose logs -f
) else if "%choice%"=="6" (
    echo Cleaning up...
    docker-compose down --volumes
    docker rmi sentiment-analysis:latest
    echo.
    echo Cleanup complete!
    pause
) else (
    echo Invalid choice!
    pause
    exit /b 1
)
