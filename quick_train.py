"""
Quick Training - Train DistilBERT model (fastest/lightest) first
"""

import pandas as pd
import torch
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("QUICK TRAINING - DistilBERT (FASTEST)")
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

# Create directory
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

# 3. Tokenize data
print("\n[3] TOKENIZING DATA")
print("-" * 80)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_dict({
    'text': texts_train,
    'label': labels_train
})
test_dataset = Dataset.from_dict({
    'text': texts_test,
    'label': labels_test
})

print("Tokenizing training data...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
print("Tokenizing test data...")
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 4. Training
print("\n[4] TRAINING MODEL")
print("-" * 80)

training_args = TrainingArguments(
    output_dir='./results/distilbert_temp',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_strategy='no',
    eval_strategy='epoch',
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Training...")
trainer.train()

# 5. Evaluate
print("\n[5] EVALUATING MODEL")
print("-" * 80)

eval_results = trainer.evaluate()
print(f"Final Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Final Precision: {eval_results['eval_precision']:.4f}")
print(f"Final Recall: {eval_results['eval_recall']:.4f}")
print(f"Final F1-Score: {eval_results['eval_f1']:.4f}")

# 6. Save model
print("\n[6] SAVING MODEL")
print("-" * 80)

model_save_path = './sentiment_models/DistilBERT'
print(f"Saving to: {model_save_path}")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save metrics
metrics = {
    'model': 'DistilBERT',
    'accuracy': float(eval_results['eval_accuracy']),
    'precision': float(eval_results['eval_precision']),
    'recall': float(eval_results['eval_recall']),
    'f1': float(eval_results['eval_f1']),
    'training_samples': len(texts_train),
    'test_samples': len(texts_test),
    'device': device
}

with open('./results/distilbert_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel saved to: {model_save_path}")
print(f"Metrics saved to: ./results/distilbert_metrics.json")
