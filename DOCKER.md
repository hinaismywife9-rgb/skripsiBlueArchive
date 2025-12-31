# 🐳 Docker Setup Guide - Sentiment Analysis

Complete Docker setup untuk menjalankan seluruh sentiment analysis pipeline dengan Streamlit dashboard.

---

## 📋 Prerequisites

### Windows
- **Docker Desktop for Windows** (https://www.docker.com/products/docker-desktop)
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### macOS
- **Docker Desktop for Mac** (https://www.docker.com/products/docker-desktop)
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### Linux
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
```

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Windows (Easiest)

1. **Open Command Prompt** in project directory
2. **Run:**
   ```cmd
   docker-run.bat
   ```
3. **Select option 2** (Run with Docker Compose)
4. **Wait for setup** (2-3 minutes first time)
5. **Open browser** → http://localhost:8501

### Option 2: macOS/Linux

1. **Open Terminal** in project directory
2. **Run:**
   ```bash
   chmod +x docker-run.sh
   ./docker-run.sh
   ```
3. **Select option 2** (Run with Docker Compose)
4. **Wait for setup** (2-3 minutes first time)
5. **Open browser** → http://localhost:8501

### Option 3: Manual Docker Commands

```bash
# Build image
docker build -t sentiment-analysis:latest .

# Run with Docker Compose
docker-compose up --build

# Or run single container
docker run -it --rm \
  -p 8501:8501 \
  -v $(pwd)/sentiment_models:/app/sentiment_models \
  -v $(pwd)/results:/app/results \
  sentiment-analysis:latest
```

---

## 📊 Dashboard Access

Once running, open your browser:
- **URL**: http://localhost:8501
- **Port**: 8501

You'll see the Streamlit dashboard with 6 pages:
1. 📈 **Overview** - Project summary
2. 🤖 **Model Training** - Training results
3. 🧪 **Test Model** - Single model testing
4. 📊 **Model Comparison** - Compare all models
5. 🎯 **Predictions** - Batch predictions & ensemble
6. 📉 **Metrics Analysis** - Advanced metrics

---

## 🐳 Docker Architecture

```
┌─────────────────────────────────┐
│   sentiment-analysis Container  │
├─────────────────────────────────┤
│ • Python 3.9                    │
│ • PyTorch                       │
│ • Transformers                  │
│ • Streamlit Dashboard           │
│ • Jupyter Support               │
└─────────────────────────────────┘
         │
         ├─ Port 8501 (Streamlit)
         │
         └─ Volumes
            ├─ sentiment_models/
            ├─ results/
            ├─ logs/
            └─ data/
```

---

## 📁 Volume Mounts

Container automatically mounts these directories:

| Container Path | Host Path | Purpose |
|---|---|---|
| `/app/sentiment_models` | `./sentiment_models` | Trained models |
| `/app/results` | `./results` | Training results |
| `/app/logs` | `./logs` | Training logs |
| `/app/data` | `./data` | Data files |

This means:
- ✅ Models persist after container stops
- ✅ Results are saved locally
- ✅ Can access files from host machine

---

## 🎯 Common Commands

### Start Container
```bash
# Windows
docker-run.bat
# Then select option 2

# macOS/Linux
./docker-run.sh
# Then select option 2

# Or direct Docker Compose
docker-compose up --build
```

### Stop Container
```bash
# Windows
docker-run.bat
# Then select option 4

# macOS/Linux
./docker-run.sh
# Then select option 4

# Or
Ctrl+C (in terminal)
# Then
docker-compose down
```

### View Logs
```bash
# Windows
docker-run.bat
# Then select option 5

# macOS/Linux
./docker-run.sh
# Then select option 5

# Or
docker-compose logs -f
```

### Execute Commands in Running Container
```bash
# Get container ID
docker ps

# Execute command
docker exec -it <container_id> python check_sentiment.py

# Or bash
docker exec -it <container_id> bash
```

---

## 🔧 Customization

### Modify Port
Edit `docker-compose.yml`:
```yaml
ports:
  - "8501:8501"  # Change first number to your desired port
```

Then restart:
```bash
docker-compose up --build
```

### Modify Python Packages
Edit `requirements.txt` and rebuild:
```bash
docker build -t sentiment-analysis:latest .
docker-compose up --build
```

### Modify Streamlit Settings
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#1f77b4"
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8501
kill -9 <PID>

# Then change port in docker-compose.yml
```

### Out of Memory
```bash
# Increase Docker memory allocation
# Docker Desktop → Settings → Resources → Memory (increase to 8GB)
```

### Volume Mount Issues
```bash
# Make sure paths exist
mkdir sentiment_models results logs data

# Check permissions
ls -la
```

### GPU Support
For GPU support, modify Dockerfile:
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu22.04

# ... rest of Dockerfile
```

Then rebuild with GPU support.

---

## 📊 Workflow

### Step 1: Build & Run Container
```bash
# Windows
docker-run.bat  # Select option 2

# macOS/Linux
./docker-run.sh  # Select option 2
```

### Step 2: Wait for Setup
```
Container starts → Installs dependencies → Starts Streamlit
Estimated time: 3-5 minutes
```

### Step 3: Open Dashboard
Open browser: http://localhost:8501

### Step 4: View Overview
Navigate to "📈 Overview" page to see project status

### Step 5: Train Models (if not already trained)
In terminal, inside container:
```bash
python train_transformer_models.py
```

Or create script in dashboard to run training.

### Step 6: Test & Evaluate
Use dashboard pages:
- Test individual models
- Compare all models
- Make batch predictions
- View advanced metrics

---

## 📈 Dashboard Features

### Page 1: Overview
- Project summary
- Data statistics
- Model information

### Page 2: Model Training
- Training status
- Performance metrics
- Comparison charts
- Visualizations

### Page 3: Test Model
- Select model
- Single text prediction
- Batch CSV processing
- Download results

### Page 4: Model Comparison
- Side-by-side metrics
- Bar charts
- Radar chart
- Performance ranking

### Page 5: Predictions
- Single model prediction
- Ensemble voting
- Majority voting
- Confidence-based voting

### Page 6: Metrics Analysis
- Detailed metrics
- Performance summary
- Export results (CSV/JSON)

---

## 🔄 Development Workflow

### Add New Python Package
1. Edit `requirements.txt`
2. Rebuild container:
   ```bash
   docker build -t sentiment-analysis:latest .
   ```
3. Restart:
   ```bash
   docker-compose up --build
   ```

### Modify Dashboard
1. Edit `dashboard.py`
2. Streamlit auto-reloads on save
3. Check browser for changes

### Debug Inside Container
```bash
# Open bash in running container
docker exec -it sentiment-analysis-app bash

# Run Python commands
python
>>> from sentiment_utils import SentimentAnalyzer
>>> analyzer = SentimentAnalyzer('./sentiment_models/RoBERTa')
>>> analyzer.predict("Test text")
```

---

## 🧹 Cleanup

### Remove Containers
```bash
docker-compose down
```

### Remove Images
```bash
docker rmi sentiment-analysis:latest
```

### Remove All (including volumes)
```bash
docker-compose down --volumes
```

### Free up space
```bash
docker system prune -a
```

---

## 📊 Monitoring

### Check Container Status
```bash
docker ps
```

### View CPU/Memory Usage
```bash
docker stats sentiment-analysis-app
```

### View Container Logs
```bash
docker logs sentiment-analysis-app
docker logs -f sentiment-analysis-app  # Follow logs
```

---

## 🔐 Security Notes

- ✅ Container runs with default user (non-root recommended)
- ✅ Volumes mounted as read-write for development
- ✅ No exposed databases or API keys in Dockerfile
- ✅ Local-only by default (localhost:8501)

For production:
- Use environment variables for config
- Add authentication to Streamlit
- Use reverse proxy (Nginx)
- Enable SSL/TLS

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## ✅ Checklist

- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] Project files downloaded
- [ ] requirements.txt updated
- [ ] Dockerfile created
- [ ] docker-compose.yml created
- [ ] Run docker-compose up
- [ ] Dashboard accessible at localhost:8501
- [ ] Models trained or loaded
- [ ] Dashboard pages working

---

## 🎉 You're Ready!

Your sentiment analysis application is now containerized and ready to run!

**Next Steps:**
1. Run Docker container: `docker-compose up --build`
2. Wait for setup to complete
3. Open http://localhost:8501
4. Start analyzing sentiment! 🚀

---

**Happy Analyzing! 📊**
