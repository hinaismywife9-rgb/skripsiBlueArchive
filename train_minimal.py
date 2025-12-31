"""
Minimal Training Script - Avoid MKL Threading Issues
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)

import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import json

print("=" * 80)
print("SENTIMENT ANALYSIS - 5 MODEL TRAINING")
print("=" * 80)

# Load data
print("\n[1] DATA")
df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
print(f"✓ Loaded {len(df)} samples")

# Prepare
texts = df['text'].tolist()
labels = df['sentiment'].tolist()
label2id = {'negative': 0, 'positive': 1}
id2label = {v: k for k, v in label2id.items()}
numeric_labels = [label2id[l] for l in labels]

# Split
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
)
print(f"✓ Train: {len(texts_train)}, Test: {len(texts_test)}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

# Models
MODELS = {
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'DistilBERT': 'distilbert-base-uncased',
    'ALBERT': 'albert-base-v2',
    'XLNET': 'xlnet-base-cased'
}

results = {}

print("\n[2] TRAINING")
print("-" * 80)

for model_name, model_path in MODELS.items():
    print(f"\n>>> {model_name}")
    try:
        # Tokenize
        print("  Tokenizing...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        train_enc = tokenizer(texts_train, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        test_enc = tokenizer(texts_test, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        # Model
        print("  Loading...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, id2label=id2label, label2id=label2id
        ).to(device)
        
        # Data loaders
        train_ds = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(labels_train))
        test_ds = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], torch.tensor(labels_test))
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)  # Smaller batch
        test_loader = DataLoader(test_ds, batch_size=8)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        # Train
        print("  Training (3 epochs)...")
        for epoch in range(3):
            model.train()
            epoch_loss = 0
            
            for idx, batch in enumerate(train_loader):
                input_ids, attn_mask, labels = [x.to(device) for x in batch]
                
                optimizer.zero_grad()
                out = model(input_ids, attention_mask=attn_mask)
                loss = loss_fn(out.logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                if (idx + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1} Step {idx+1}: Loss={loss.item():.4f}")
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Eval
        print("  Evaluating...")
        model.eval()
        preds = []
        true = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attn_mask, labels = [x.to(device) for x in batch]
                out = model(input_ids, attention_mask=attn_mask)
                pred = torch.argmax(out.logits, dim=1).cpu().numpy()
                preds.extend(pred)
                true.extend(labels.numpy())
        
        # Metrics
        acc = accuracy_score(true, preds)
        prec = precision_score(true, preds, average='weighted', zero_division=0)
        rec = recall_score(true, preds, average='weighted', zero_division=0)
        f1 = f1_score(true, preds, average='weighted', zero_division=0)
        
        # Save
        os.makedirs(f'sentiment_models/{model_name}', exist_ok=True)
        model.save_pretrained(f'sentiment_models/{model_name}')
        tokenizer.save_pretrained(f'sentiment_models/{model_name}')
        
        results[model_name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
        print(f"  ✓ Acc={acc:.4f}, F1={f1:.4f}")
    
    except Exception as e:
        print(f"  ✗ {e}")

# Results
print("\n[3] RESULTS")
print("=" * 80)

df_res = pd.DataFrame([{'Model': k, 'Accuracy': v['acc'], 'Precision': v['prec'], 'Recall': v['rec'], 'F1': v['f1']} 
                       for k, v in results.items()])
df_res = df_res.sort_values('F1', ascending=False)
print(df_res.to_string(index=False))

df_res.to_csv('model_performance_comparison.csv', index=False)
with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✓ TRAINING COMPLETE!")
if len(results) > 0:
    best = df_res.iloc[0]
    print(f"✓ Best: {best['Model']} (F1={best['F1']:.4f})")
print("=" * 80)
