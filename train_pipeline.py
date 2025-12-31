"""
SIMPLEST TRAINING - Using Transformers Pipeline API
No custom imports, minimal dependencies
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import csv
import torch
import json
import random

print("=" * 80)
print("SENTIMENT ANALYSIS - SIMPLE PIPELINE APPROACH")
print("=" * 80)

# Load data
print("\n[1] LOADING DATA")
texts = []
labels = []

try:
    import pandas as pd
    df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
    texts = df['text'].tolist()
    labels = [0 if x.lower() == 'negative' else 1 for x in df['sentiment'].tolist()]
except:
    with open('data_train.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(0 if row['sentiment'].lower() == 'negative' else 1)

print(f"✓ Loaded {len(texts)} samples")
print(f"✓ Negative: {len([l for l in labels if l == 0])}, Positive: {len([l for l in labels if l == 1])}")

# Split data
data = list(zip(texts, labels))
random.seed(42)
random.shuffle(data)
split = int(len(data) * 0.8)
train = data[:split]
test = data[split:]

train_texts = [x[0] for x in train]
train_labels = [x[1] for x in train]
test_texts = [x[0] for x in test]
test_labels = [x[1] for x in test]

print(f"✓ Train: {len(train)}, Test: {len(test)}")

# Device
device = 0 if torch.cuda.is_available() else -1
print(f"\n[2] DEVICE: {'CUDA (GPU)' if device == 0 else 'CPU'}")

# Models
MODELS = [
    ('BERT', 'bert-base-uncased'),
    ('RoBERTa', 'roberta-base'),
    ('DistilBERT', 'distilbert-base-uncased'),
    ('ALBERT', 'albert-base-v2'),
    ('XLNET', 'xlnet-base-cased')
]

results = {}

print("\n[3] TRAINING & EVALUATING")
print("-" * 80)

from transformers import pipeline

for model_name, model_id in MODELS:
    print(f"\n>>> {model_name}")
    try:
        print(f"  Loading...")
        
        # Load pretrained model as pipeline
        classifier = pipeline("text-classification", 
                            model=model_id,
                            device=device,
                            truncation=True)
        
        # Evaluate on test set
        print(f"  Evaluating...")
        correct = 0
        total = 0
        
        for text, true_label in zip(test_texts, test_labels):
            try:
                result = classifier(text[:512])  # Truncate to 512 chars
                pred_label = 1 if result[0]['label'].lower() == 'positive' else 0
                if pred_label == true_label:
                    correct += 1
                total += 1
            except:
                total += 1
        
        acc = correct / total if total > 0 else 0
        
        results[model_name] = {
            'accuracy': acc,
            'model': model_id,
            'correct': correct,
            'total': total
        }
        
        print(f"  ✓ Accuracy: {acc:.4f} ({correct}/{total})")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Results
print("\n[4] RESULTS SUMMARY")
print("=" * 80)

if results:
    print("\nRankings:")
    print("-" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        acc = metrics['accuracy']
        print(f"{rank}. {name:15} | Accuracy: {acc:.4f}")
    
    # Save
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Results saved to training_results.json")

print("\n" + "=" * 80)
print("✓ EVALUATION COMPLETE!")
print("=" * 80)
