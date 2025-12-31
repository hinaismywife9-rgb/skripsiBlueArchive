"""
Ultra-Simple Training - No Pandas Excel dependency
Just train on pre-prepared text files
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)

# Load simple CSV instead
import csv

print("=" * 80)
print("SENTIMENT ANALYSIS TRAINING - 5 MODELS")
print("=" * 80)

# Convert Excel to CSV first if needed
import subprocess
import sys

try:
    # Try to convert Excel to CSV using pandas
    try:
        import pandas as pd
        df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
        df.to_csv('training_data.csv', index=False)
        print("✓ Converted Excel to CSV")
    except:
        print("⚠ Using existing training_data.csv")

    # Load data from CSV
    texts = []
    labels = []
    with open('training_data.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(0 if row['sentiment'] == 'negative' else 1)
    
    print(f"\n[1] DATA LOADED")
    print(f"✓ Total samples: {len(texts)}")
    print(f"✓ Labels: {len([l for l in labels if l == 0])} negative, {len([l for l in labels if l == 1])} positive")
    
    # Split manually
    from sklearn.model_selection import train_test_split
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"✓ Train: {len(texts_train)}, Test: {len(texts_test)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2] DEVICE: {device}")
    
    # Models to train
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
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import json
    
    results = {}
    
    print("\n[3] TRAINING MODELS")
    print("-" * 80)
    
    for model_name, model_path in MODELS.items():
        print(f"\n>>> {model_name}")
        try:
            # Load
            print("  Loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=2
            ).to(device)
            
            # Tokenize
            print("  Tokenizing...")
            train_enc = tokenizer(texts_train, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            test_enc = tokenizer(texts_test, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            
            # Datasets
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
            
            # Optimize
            print("  Training (3 epochs)...")
            opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
            loss_fn = nn.CrossEntropyLoss()
            
            for epoch in range(3):
                model.train()
                tot_loss = 0
                for step, batch in enumerate(train_loader):
                    ids, mask, lbl = [x.to(device) for x in batch]
                    opt.zero_grad()
                    out = model(ids, attention_mask=mask)
                    loss = loss_fn(out.logits, lbl)
                    loss.backward()
                    opt.step()
                    tot_loss += loss.item()
                    if (step + 1) % 20 == 0:
                        print(f"    E{epoch+1}S{step+1}: L={loss.item():.4f}")
            
            # Eval
            print("  Evaluating...")
            model.eval()
            preds, true = [], []
            with torch.no_grad():
                for batch in test_loader:
                    ids, mask, lbl = [x.to(device) for x in batch]
                    out = model(ids, attention_mask=mask)
                    p = torch.argmax(out.logits, dim=1).cpu().numpy()
                    preds.extend(p)
                    true.extend(lbl.numpy())
            
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
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Results
    print("\n[4] RESULTS")
    print("=" * 80)
    
    # Create results table
    print("\nModel Rankings (by F1-Score):")
    print("-" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for rank, (model, metrics) in enumerate(sorted_results, 1):
        print(f"{rank}. {model:15} | Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f}")
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Saved: training_results.json")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
