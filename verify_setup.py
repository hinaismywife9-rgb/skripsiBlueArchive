"""
Simple test script to verify models and dashboard setup
Run this before starting Streamlit dashboard
"""

import os
import sys
import torch
from pathlib import Path

print("=" * 60)
print("🔍 SENTIMENT ANALYSIS DASHBOARD - VERIFICATION TEST")
print("=" * 60)

# Check Python version
print(f"\n✓ Python {sys.version.split()[0]}")

# Check PyTorch
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")

# Check models
print("\n📁 Checking models...")
models_dir = Path("./sentiment_models")
model_names = ['BERT', 'DistilBERT', 'RoBERTa', 'ALBERT', 'XLNET']

models_found = 0
for model_name in model_names:
    model_path = models_dir / model_name
    if model_path.exists():
        print(f"  ✓ {model_name}")
        models_found += 1
    else:
        print(f"  ✗ {model_name} - NOT FOUND")

print(f"\n{models_found}/{len(model_names)} models found")

if models_found < len(model_names):
    print("\n⚠️  Warning: Not all models found. Train them first!")
    sys.exit(1)

# Check required packages
print("\n📦 Checking packages...")
required_packages = [
    ('transformers', 'Transformers'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
]

missing_packages = []
for package, display_name in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {display_name}")
    except ImportError:
        print(f"  ✗ {display_name} - NOT INSTALLED")
        missing_packages.append(package)

# Check Streamlit
try:
    import streamlit
    print(f"  ✓ Streamlit {streamlit.__version__}")
except ImportError:
    print(f"  ✗ Streamlit - NOT INSTALLED")
    missing_packages.append('streamlit')

# Check WordCloud
try:
    import wordcloud
    print(f"  ✓ WordCloud")
except ImportError:
    print(f"  ✗ WordCloud - NOT INSTALLED")
    missing_packages.append('wordcloud')

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Try loading one model
print("\n🔧 Testing model loading...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_path = "./sentiment_models/DistilBERT"
    print(f"  Loading {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    print("  ✓ DistilBERT loaded successfully")
    
    # Quick inference test
    text = "This is great!"
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"  ✓ Test prediction: '{text}' → {sentiment} ({confidence:.2%})")
    
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    sys.exit(1)

# Check dashboard files
print("\n📄 Checking dashboard files...")
dashboard_files = ['app.py', 'DASHBOARD_DOCUMENTATION.md', 'DASHBOARD_QUICKSTART.md']

for filename in dashboard_files:
    if os.path.exists(filename):
        size_kb = os.path.getsize(filename) / 1024
        print(f"  ✓ {filename} ({size_kb:.1f} KB)")
    else:
        print(f"  ✗ {filename} - NOT FOUND")

# Summary
print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\n🚀 You can now run the dashboard:")
print("\n   streamlit run app.py")
print("\nOr double-click: run_streamlit.bat")
print("\n📊 Dashboard will open at: http://localhost:8501")
print("=" * 60)
