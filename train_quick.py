"""
Quick Sentiment Analysis Training - No Complex Dependencies
Menggunakan approach yang lebih simple untuk menghindari MKL threading issues
"""

import os
import sys

# Fix MKL issues FIRST
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Now import everything else
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

print("=" * 80)
print("QUICK SENTIMENT ANALYSIS TRAINING - 5 MODELS")
print("=" * 80)

# Load data
print("\n[1] Loading Data...")
df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
print(f"✓ Loaded {len(df)} samples")
print(f"  Distribution: {df['sentiment'].value_counts().to_dict()}")

# Import torch AFTER setting env vars
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Prepare data
texts = df['text'].tolist()
labels = df['sentiment'].tolist()

label2id = {'negative': 0, 'positive': 1}
id2label = {0: 'negative', 1: 'positive'}
numeric_labels = [label2id[label] for label in labels]

texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
)

print(f"✓ Train: {len(texts_train)}, Test: {len(texts_test)}")

# Models
MODELS = {
    'BERT': 'bert-base-uncased',
    'DistilBERT': 'distilbert-base-uncased',
    'RoBERTa': 'roberta-base',
    'ALBERT': 'albert-base-v2',
    'XLNET': 'xlnet-base-cased'
}

def train_model(name, model_path):
    """Train a single model"""
    print(f"\n>>> Training {name}...")
    
    try:
        # Load tokenizer
        print(f"    Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tokenize
        print(f"    Tokenizing data...")
        train_enc = tokenizer(texts_train, padding=True, truncation=True, max_length=128, return_tensors='pt')
        test_enc = tokenizer(texts_test, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        # Load model
        print(f"    Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, id2label=id2label, label2id=label2id
        ).to(device)
        
        # Setup training
        train_ds = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(labels_train))
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training
        print(f"    Training (3 epochs)...")
        for epoch in range(3):
            model.train()
            total_loss = 0
            count = 0
            
            for batch in train_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
            
            print(f"      Epoch {epoch+1} Loss: {total_loss/count:.4f}")
        
        # Evaluate
        print(f"    Evaluating...")
        model.eval()
        test_ds = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], torch.tensor(labels_test))
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2]
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.numpy())
        
        # Metrics
        acc = accuracy_score(true_labels, predictions)
        prec = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        rec = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Save
        os.makedirs(f'./sentiment_models/{name}', exist_ok=True)
        model.save_pretrained(f'./sentiment_models/{name}')
        tokenizer.save_pretrained(f'./sentiment_models/{name}')
        
        print(f"    ✓ {name} Done! Acc: {acc:.4f}, F1: {f1:.4f}")
        
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None

# Train all models
print("\n[2] Training Models")
print("-" * 80)

results = {}
for name, path in MODELS.items():
    result = train_model(name, path)
    if result:
        results[name] = result

# Summary
print("\n[3] Results Summary")
print("=" * 80)

if results:
    df_results = pd.DataFrame([
        {'Model': k, 'Accuracy': v['acc'], 'Precision': v['prec'], 'Recall': v['rec'], 'F1': v['f1']}
        for k, v in results.items()
    ])
    df_results = df_results.sort_values('F1', ascending=False)
    print("\n" + df_results.to_string(index=False))
    
    # Save results
    df_results.to_csv('model_performance_comparison.csv', index=False)
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Results saved to:")
    print("  - model_performance_comparison.csv")
    print("  - training_results.json")
    
    best = df_results.iloc[0]
    print(f"\n✓ BEST MODEL: {best['Model']}")
    print(f"  F1-Score: {best['F1']:.4f}")
    print(f"  Location: ./sentiment_models/{best['Model']}/")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
