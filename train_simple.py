"""
Simple Training Script untuk 5 Transformer Models
Tanpa menggunakan Trainer untuk menghindari masalah kompatibilitas
"""

import os
import sys

# Set environment variables BEFORE importing numpy/torch to avoid Intel MKL issues
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Models configuration
MODELS_CONFIG = {
    'BERT': {
        'model_name': 'bert-base-uncased',
        'description': 'Bidirectional Encoder Representations from Transformers',
        'best_for': 'General purpose NLP'
    },
    'DistilBERT': {
        'model_name': 'distilbert-base-uncased',
        'description': 'Distilled BERT (40% smaller, 60% faster)',
        'best_for': 'Fast inference with minimal performance loss'
    },
    'RoBERTa': {
        'model_name': 'roberta-base',
        'description': 'Robustly Optimized BERT Pretraining',
        'best_for': 'Best for sentiment analysis tasks'
    },
    'ALBERT': {
        'model_name': 'albert-base-v2',
        'description': 'A Lite BERT - memory efficient',
        'best_for': 'Mobile/edge deployment'
    },
    'XLNET': {
        'model_name': 'xlnet-base-cased',
        'description': 'Autoregressive pretraining',
        'best_for': 'Complex context understanding'
    }
}

print("=" * 80)
print("TRANSFORMER MODELS UNTUK SENTIMENT ANALYSIS - SIMPLE TRAINING")
print("=" * 80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# 1. Load and prepare data
print("\n[1] LOADING DATA")
print("-" * 80)

df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
print(f"Total samples: {len(df)}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

texts = df['text'].tolist()
labels = df['sentiment'].tolist()

# Convert labels
label2id = {'negative': 0, 'positive': 1}
id2label = {0: 'negative', 1: 'positive'}
numeric_labels = [label2id[label] for label in labels]

# Split data
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
)

print(f"Training samples: {len(texts_train)}")
print(f"Testing samples: {len(texts_test)}")

# 2. Training function
def train_model(model_name, model_config, texts_train, labels_train, texts_test, labels_test, device):
    print(f"\n>>> Training {model_name}...")
    
    try:
        # Load tokenizer and model
        print("   Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config['model_name'],
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        ).to(device)
        
        # Tokenize data
        print("   Tokenizing data...")
        train_encodings = tokenizer(
            texts_train, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        )
        test_encodings = tokenizer(
            texts_test, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(labels_train)
        )
        test_dataset = TensorDataset(
            test_encodings['input_ids'],
            test_encodings['attention_mask'],
            torch.tensor(labels_test)
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()
        num_epochs = 3
        
        print(f"   Training for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            print(f"   Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}")
        
        # Evaluation
        print("   Evaluating on test set...")
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                predictions.extend(batch_predictions)
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Save model
        os.makedirs(f'./sentiment_models/{model_name}', exist_ok=True)
        model.save_pretrained(f'./sentiment_models/{model_name}')
        tokenizer.save_pretrained(f'./sentiment_models/{model_name}')
        
        print(f"   ✓ {model_name} completed!")
        print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_name': model_config['model_name'],
            'description': model_config['description'],
            'model_path': f'./sentiment_models/{model_name}'
        }
        
    except Exception as e:
        print(f"   ✗ Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

# 3. Train all models
print("\n[2] MODEL INFORMATION")
print("-" * 80)
for idx, (model_name, config) in enumerate(MODELS_CONFIG.items(), 1):
    print(f"\n{idx}. {model_name}")
    print(f"   Model: {config['model_name']}")
    print(f"   Description: {config['description']}")
    print(f"   Best for: {config['best_for']}")

print("\n[3] TRAINING MODELS")
print("-" * 80)

results = {}
for model_name, config in MODELS_CONFIG.items():
    result = train_model(model_name, config, texts_train, labels_train, texts_test, labels_test, device)
    results[model_name] = result

# 4. Results Summary
print("\n[4] TRAINING RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame([
    {
        'Model': model_name,
        'Accuracy': results[model_name].get('accuracy', 0),
        'Precision': results[model_name].get('precision', 0),
        'Recall': results[model_name].get('recall', 0),
        'F1-Score': results[model_name].get('f1', 0),
    }
    for model_name in MODELS_CONFIG.keys()
    if 'accuracy' in results.get(model_name, {})
])

if len(results_df) > 0:
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df) + 1)
    
    print("\n" + results_df[['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

# 5. Save results
print("\n[5] SAVING RESULTS")
print("-" * 80)

# Save as JSON
with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
    print("✓ Saved: training_results.json")

# Save as CSV
if len(results_df) > 0:
    results_df.to_csv('model_performance_comparison.csv', index=False)
    print("✓ Saved: model_performance_comparison.csv")

# Save report
with open('model_training_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TRANSFORMER MODELS TRAINING REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TRAINING RESULTS\n")
    f.write("-" * 80 + "\n")
    if len(results_df) > 0:
        f.write(results_df[['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    if len(results_df) > 0:
        best_model = results_df.iloc[0]['Model']
        f.write(f"\n\n✓ BEST MODEL: {best_model}\n")
        f.write(f"  Accuracy: {results_df.iloc[0]['Accuracy']:.4f}\n")
        f.write(f"  F1-Score: {results_df.iloc[0]['F1-Score']:.4f}\n")

print("✓ Saved: model_training_report.txt")

# 6. Final summary
print("\n[6] FINAL SUMMARY")
print("=" * 80)

if len(results_df) > 0:
    best_model = results_df.iloc[0]
    print(f"\n✓ BEST MODEL: {best_model['Model']}")
    print(f"  Accuracy: {best_model['Accuracy']:.4f}")
    print(f"  Precision: {best_model['Precision']:.4f}")
    print(f"  Recall: {best_model['Recall']:.4f}")
    print(f"  F1-Score: {best_model['F1-Score']:.4f}")
    print(f"\n✓ Model saved at: ./sentiment_models/{best_model['Model']}/")
    
    print("\n\nNEXT STEPS:")
    print("1. Test models:")
    print("   python inference_sentiment.py")
    print("\n2. Use in your code:")
    print(f"   from sentiment_utils import SentimentAnalyzer")
    print(f"   analyzer = SentimentAnalyzer('./sentiment_models/{best_model['Model']}')")
    print(f"   result = analyzer.predict('Your text here')")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
