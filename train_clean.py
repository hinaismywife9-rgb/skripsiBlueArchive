"""
Super-Simple Training Script - Minimal Dependencies
No Pandas, No SKlearn - Just Pure Python + PyTorch + Transformers
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)

import csv
import random

print("=" * 80)
print("SENTIMENT ANALYSIS TRAINING - 5 MODELS (VENV VERSION)")
print("=" * 80)

# Load data
print("\n[1] LOADING DATA")
try:
    import pandas as pd
    df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
    df.to_csv('data_train.csv', index=False)
    print("✓ Converted Excel to CSV")
except:
    pass

texts = []
labels = []
with open('data_train.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['text'].strip())
        labels.append(0 if row['sentiment'].lower() == 'negative' else 1)

print(f"✓ Loaded {len(texts)} samples")
pos = sum(labels)
neg = len(labels) - pos
print(f"✓ Distribution: {neg} negative, {pos} positive")

# Manual split (80/20)
data = list(zip(texts, labels))
random.seed(42)
random.shuffle(data)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

texts_train = [x[0] for x in train_data]
labels_train = [x[1] for x in train_data]
texts_test = [x[0] for x in test_data]
labels_test = [x[1] for x in test_data]

print(f"✓ Split: {len(texts_train)} train, {len(texts_test)} test")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[2] DEVICE: {device}")

MODELS = {
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'DistilBERT': 'distilbert-base-uncased',
    'ALBERT': 'albert-base-v2',
    'XLNET': 'xlnet-base-cased'
}

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

def calc_metrics(true, pred):
    """Calculate metrics manually"""
    tp = sum(1 for t, p in zip(true, pred) if t == p and t == 1)
    tn = sum(1 for t, p in zip(true, pred) if t == p and t == 0)
    fp = sum(1 for t, p in zip(true, pred) if t != p and p == 1)
    fn = sum(1 for t, p in zip(true, pred) if t != p and p == 0)
    
    acc = (tp + tn) / len(true) if len(true) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

results = {}

print("\n[3] TRAINING")
print("-" * 80)

for model_name, model_path in MODELS.items():
    print(f"\n>>> {model_name}")
    try:
        # Load
        print("  Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(device)
        
        # Tokenize
        print("  Tokenizing...")
        train_enc = tokenizer(texts_train, max_length=128, padding='max_length', 
                            truncation=True, return_tensors='pt')
        test_enc = tokenizer(texts_test, max_length=128, padding='max_length',
                           truncation=True, return_tensors='pt')
        
        # Data
        train_ds = TensorDataset(
            train_enc['input_ids'], 
            train_enc['attention_mask'], 
            torch.tensor(labels_train)
        )
        test_ds = TensorDataset(
            test_enc['input_ids'],
            test_enc['attention_mask'],
            torch.tensor(labels_test)
        )
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=8)
        
        # Train
        print("  Training (3 epochs)...")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(3):
            model.train()
            for step, (ids, mask, lbl) in enumerate(train_loader):
                ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
                opt.zero_grad()
                out = model(ids, attention_mask=mask)
                loss = loss_fn(out.logits, lbl)
                loss.backward()
                opt.step()
                if (step + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1} Step {step+1}: Loss={loss.item():.4f}")
        
        # Eval
        print("  Evaluating...")
        model.eval()
        preds = []
        true_labels = []
        with torch.no_grad():
            for ids, mask, lbl in test_loader:
                ids, mask = ids.to(device), mask.to(device)
                out = model(ids, attention_mask=mask)
                p = torch.argmax(out.logits, dim=1).cpu().numpy().tolist()
                preds.extend(p)
                true_labels.extend(lbl.numpy().tolist())
        
        # Metrics
        metrics = calc_metrics(true_labels, preds)
        
        # Save
        os.makedirs(f'sentiment_models/{model_name}', exist_ok=True)
        model.save_pretrained(f'sentiment_models/{model_name}')
        tokenizer.save_pretrained(f'sentiment_models/{model_name}')
        
        results[model_name] = metrics
        print(f"  ✓ Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}")
    
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback
        traceback.print_exc()

# Results
print("\n[4] RESULTS")
print("=" * 80)

print("\nModel Performance Rankings:")
print("-" * 70)
sorted_r = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
for rank, (name, m) in enumerate(sorted_r, 1):
    print(f"{rank}. {name:15} | Accuracy={m['acc']:.4f} | Precision={m['prec']:.4f} | F1={m['f1']:.4f}")

# Save
import json
with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✓ Results saved to training_results.json")
print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
