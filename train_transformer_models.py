"""
5 Transformer Models untuk Sentiment Analysis
Models: BERT, DistilBERT, RoBERTa, ALBERT, dan XLNet
"""

import pandas as pd
import numpy as np
import torch
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
MODELS_CONFIG = {
    'BERT': {
        'model_name': 'bert-base-uncased',
        'description': 'Bidirectional Encoder Representations from Transformers',
        'best_for': 'General purpose NLP, balances performance and speed'
    },
    'DistilBERT': {
        'model_name': 'distilbert-base-uncased',
        'description': 'Distilled version of BERT (40% smaller, 60% faster)',
        'best_for': 'Fast inference with minimal performance loss'
    },
    'RoBERTa': {
        'model_name': 'roberta-base',
        'description': 'Robustly Optimized BERT Pretraining',
        'best_for': 'Better performance on sentiment analysis tasks'
    },
    'ALBERT': {
        'model_name': 'albert-base-v2',
        'description': 'A Lite BERT - parameter reduction technique',
        'best_for': 'Memory efficient with competitive performance'
    },
    'XLNET': {
        'model_name': 'xlnet-base-cased',
        'description': 'Autoregressive pretraining with permutation language modeling',
        'best_for': 'Complex context understanding'
    }
}

print("=" * 80)
print("TRANSFORMER MODELS UNTUK SENTIMENT ANALYSIS")
print("=" * 80)

# 1. Load data
print("\n[1] LOADING DATA")
print("-" * 80)

df = pd.read_excel('sentiment analysis BA_CLEANED.xlsx', sheet_name='Hybrid')
print(f"Total samples: {len(df)}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

# Prepare data
texts = df['text'].tolist()
labels = df['sentiment'].tolist()

# Convert labels to numeric
label2id = {'negative': 0, 'positive': 1}
id2label = {0: 'negative', 1: 'positive'}
numeric_labels = [label2id[label] for label in labels]

# Split data
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
)

print(f"\nTraining samples: {len(texts_train)}")
print(f"Testing samples: {len(texts_test)}")

# 2. Model Information
print("\n[2] MODEL INFORMATION")
print("-" * 80)

for idx, (model_name, config) in enumerate(MODELS_CONFIG.items(), 1):
    print(f"\n{idx}. {model_name}")
    print(f"   Model: {config['model_name']}")
    print(f"   Description: {config['description']}")
    print(f"   Best for: {config['best_for']}")

# 3. Training configuration
TRAINING_CONFIG = {
    'output_dir': './results',
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'logging_dir': './logs',
    'logging_steps': 100,
    'save_strategy': 'no',  # Don't save intermediate checkpoints
}

print("\n[3] TRAINING CONFIGURATION")
print("-" * 80)
for key, value in TRAINING_CONFIG.items():
    print(f"{key}: {value}")

# 4. Training all models
print("\n[4] TRAINING MODELS")
print("-" * 80)

results = {}
trained_models = {}

for model_name, config in MODELS_CONFIG.items():
    print(f"\n>>> Training {model_name}...")
    print("   Loading model and tokenizer...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Tokenize data
        def preprocess_function(texts):
            return tokenizer(texts, padding='max_length', truncation=True, max_length=128)
        
        train_encodings = preprocess_function(texts_train)
        test_encodings = preprocess_function(texts_test)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'label': labels_train
        })
        
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'label': labels_test
        })
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
        
        # Define metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
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
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{model_name}',
            num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
            per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
            per_device_eval_batch_size=TRAINING_CONFIG['per_device_eval_batch_size'],
            warmup_steps=TRAINING_CONFIG['warmup_steps'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
            logging_dir=TRAINING_CONFIG['logging_dir'],
            logging_steps=TRAINING_CONFIG['logging_steps'],
            save_strategy='no',
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train
        print(f"   Training in progress...")
        trainer.train()
        
        # Manual evaluation on test set
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i in range(0, len(test_encodings['input_ids']), 16):
                batch_input_ids = torch.tensor(test_encodings['input_ids'][i:i+16]).to(model.device)
                batch_attention_mask = torch.tensor(test_encodings['attention_mask'][i:i+16]).to(model.device)
                
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                predictions.extend(batch_predictions)
                true_labels.extend(labels_test[i:i+16])
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_name': config['model_name'],
            'description': config['description'],
            'model_path': f'./sentiment_models/{model_name}'
        }
        
        trained_models[model_name] = {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': None
        }
        
        # Save model locally
        os.makedirs(f'./sentiment_models/{model_name}', exist_ok=True)
        model.save_pretrained(f'./sentiment_models/{model_name}')
        tokenizer.save_pretrained(f'./sentiment_models/{model_name}')
        
        print(f"   ✓ {model_name} training completed!")
        print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"   F1-Score: {eval_results['eval_f1']:.4f}")
        
    except Exception as e:
        print(f"   ✗ Error training {model_name}: {str(e)}")
        results[model_name] = {'error': str(e)}

# 5. Results Summary
print("\n[5] TRAINING RESULTS SUMMARY")
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

results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
results_df['Rank'] = range(1, len(results_df) + 1)

print("\n" + results_df[['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

# 6. Save results
print("\n[6] SAVING RESULTS")
print("-" * 80)

# Save results as JSON
with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
    print("✓ Saved: training_results.json")

# Save results as CSV
results_df.to_csv('model_performance_comparison.csv', index=False)
print("✓ Saved: model_performance_comparison.csv")

# Save summary report
with open('model_training_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TRANSFORMER MODELS TRAINING REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("MODELS OVERVIEW\n")
    f.write("-" * 80 + "\n")
    for idx, (model_name, config) in enumerate(MODELS_CONFIG.items(), 1):
        f.write(f"\n{idx}. {model_name}\n")
        f.write(f"   Base Model: {config['model_name']}\n")
        f.write(f"   Description: {config['description']}\n")
        f.write(f"   Best for: {config['best_for']}\n")
    
    f.write("\n\nTRAINING RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(results_df[['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    f.write("\n\n\nRECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    if len(results_df) > 0:
        best_model = results_df.iloc[0]['Model']
        f.write(f"✓ Best Overall: {best_model} (F1-Score: {results_df.iloc[0]['F1-Score']:.4f})\n")
        if len(results_df) > 1:
            f.write(f"✓ Good Alternative: {results_df.iloc[1]['Model']} (F1-Score: {results_df.iloc[1]['F1-Score']:.4f})\n")

print("✓ Saved: model_training_report.txt")

# 7. Final recommendations
print("\n[7] RECOMMENDATIONS")
print("=" * 80)

if len(results_df) > 0:
    best_model = results_df.iloc[0]
    print(f"\n✓ BEST MODEL: {best_model['Model']}")
    print(f"  - Accuracy: {best_model['Accuracy']:.4f}")
    print(f"  - F1-Score: {best_model['F1-Score']:.4f}")
    print(f"  - Location: ./sentiment_models/{best_model['Model']}/")
    
    print(f"\n✓ RANKING:")
    for idx, row in results_df.iterrows():
        print(f"  {idx+1}. {row['Model']:15s} - F1: {row['F1-Score']:.4f} | Accuracy: {row['Accuracy']:.4f}")

print("\n✓ NEXT STEPS:")
print("  1. Review model performance in 'model_performance_comparison.csv'")
print("  2. Choose best model from ranking above")
print("  3. Use model for inference with 'inference_sentiment.py'")
print("  4. Deploy to production")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
