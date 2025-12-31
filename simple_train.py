"""
SIMPLE TRAINING - Manual training loop (no Trainer API needed)
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

print("=" * 80)
print("SIMPLE TRAINING - DistilBERT (Manual Training Loop)")
print("=" * 80)

# 1. Load data
print("\n[1] LOADING DATA")
print("-" * 80)

df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
print(f"Total samples: {len(df)}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

texts = df['text'].tolist()
labels = df['sentiment'].tolist()

label2id = {'negative': 0, 'positive': 1}
id2label = {0: 'negative', 1: 'positive'}
numeric_labels = [label2id[label] for label in labels]

texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
)

print(f"Training samples: {len(texts_train)}")
print(f"Testing samples: {len(texts_test)}")

# 2. Setup
print("\n[2] SETTING UP MODEL")
print("-" * 80)

MODEL_NAME = 'distilbert-base-uncased'
print(f"Model: {MODEL_NAME}")

os.makedirs('./sentiment_models/DistilBERT', exist_ok=True)

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

# 3. Create dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 4. Create dataloaders
print("\n[3] CREATING DATALOADERS")
print("-" * 80)

train_dataset = SentimentDataset(texts_train, labels_train, tokenizer)
test_dataset = SentimentDataset(texts_test, labels_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Testing batches: {len(test_loader)}")

# 5. Training
print("\n[4] TRAINING MODEL")
print("-" * 80)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 2

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Training
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    
    # Evaluation
    print("Evaluating...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

# 6. Final evaluation
print("\n[5] FINAL EVALUATION")
print("-" * 80)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Final Evaluation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

print(f"\nFinal Accuracy:  {accuracy:.4f}")
print(f"Final Precision: {precision:.4f}")
print(f"Final Recall:    {recall:.4f}")
print(f"Final F1-Score:  {f1:.4f}")

# 7. Save model
print("\n[6] SAVING MODEL")
print("-" * 80)

model_save_path = './sentiment_models/DistilBERT'
print(f"Saving to: {model_save_path}")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save metrics
metrics = {
    'model': 'DistilBERT',
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'training_samples': len(texts_train),
    'test_samples': len(texts_test),
    'device': device,
    'epochs': num_epochs
}

with open('./results/distilbert_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel saved to: {model_save_path}")
print(f"Metrics saved to: ./results/distilbert_metrics.json")
print(f"\nModel Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
