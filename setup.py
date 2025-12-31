"""
Quick Setup dan Installation Script
"""

import os
import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("SENTIMENT ANALYSIS TRANSFORMER MODELS - SETUP")
print("=" * 80)

# Check Python version
print("\n[1] CHECKING PYTHON VERSION")
print("-" * 80)

if sys.version_info < (3, 8):
    print("✗ Python 3.8 or higher is required!")
    sys.exit(1)
else:
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Check for GPU
print("\n[2] CHECKING GPU AVAILABILITY")
print("-" * 80)

try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print("  Training will be FAST with GPU!")
    else:
        print("⚠ GPU not detected. Training will use CPU (slower)")
        print("  To enable GPU support, install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
except:
    print("⚠ PyTorch not installed yet. Will install in next step.")

# Install requirements
print("\n[3] INSTALLING REQUIREMENTS")
print("-" * 80)

try:
    print("Installing packages from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ All requirements installed successfully!")
except Exception as e:
    print(f"✗ Error installing requirements: {str(e)}")
    sys.exit(1)

# Create necessary directories
print("\n[4] CREATING DIRECTORIES")
print("-" * 80)

directories = [
    './sentiment_models',
    './results',
    './logs',
    './data'
]

for dir_path in directories:
    Path(dir_path).mkdir(exist_ok=True)
    print(f"✓ {dir_path}/")

# Summary
print("\n[5] SETUP SUMMARY")
print("=" * 80)

print("\n✓ SETUP COMPLETE!")
print("\nNext steps:")
print("1. Prepare data:")
print("   python balance_sentiment.py")
print("\n2. Train all 5 models:")
print("   python train_transformer_models.py")
print("\n3. Test models:")
print("   python inference_sentiment.py")
print("\n4. Use models:")
print("   python sentiment_utils.py")

print("\nFor more information, see README.md")
print("\n" + "=" * 80)
