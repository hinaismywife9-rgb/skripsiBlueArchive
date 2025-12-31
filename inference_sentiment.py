"""
Sentiment Analysis Inference Script
Menggunakan model transformer yang sudah dilatih
"""

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import json

print("=" * 80)
print("SENTIMENT ANALYSIS INFERENCE")
print("=" * 80)

# Load trained models
MODELS = {
    'BERT': './sentiment_models/BERT',
    'DistilBERT': './sentiment_models/DistilBERT',
    'RoBERTa': './sentiment_models/RoBERTa',
    'ALBERT': './sentiment_models/ALBERT',
    'XLNET': './sentiment_models/XLNET',
}

def load_model(model_path, model_name):
    """Load a trained model"""
    try:
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
            print(f"✓ Loaded {model_name}")
            return pipe
        else:
            print(f"✗ Model path not found: {model_path}")
            return None
    except Exception as e:
        print(f"✗ Error loading {model_name}: {str(e)}")
        return None

def predict_sentiment(text, pipeline_model):
    """Predict sentiment for a single text"""
    try:
        result = pipeline_model(text, truncation=True)[0]
        return {
            'text': text,
            'label': result['label'],
            'score': result['score']
        }
    except Exception as e:
        return {
            'text': text,
            'label': 'ERROR',
            'score': 0.0,
            'error': str(e)
        }

def predict_batch(texts, pipeline_model, batch_size=32):
    """Predict sentiment for multiple texts"""
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            pred = predict_sentiment(text, pipeline_model)
            predictions.append(pred)
    return predictions

# Example texts for testing
EXAMPLE_TEXTS = [
    "I love this product! It's amazing!",
    "This is terrible and I hate it.",
    "It's okay, not great but not bad either.",
    "Absolutely fantastic experience!",
    "Worst purchase ever made.",
    "Best game ever, so much fun!",
    "Boring and disappointing.",
]

print("\n[1] LOADING MODELS")
print("-" * 80)

loaded_models = {}
for model_name, model_path in MODELS.items():
    pipe = load_model(model_path, model_name)
    if pipe:
        loaded_models[model_name] = pipe

print(f"\nSuccessfully loaded {len(loaded_models)}/{len(MODELS)} models")

if loaded_models:
    print("\n[2] TESTING WITH EXAMPLE TEXTS")
    print("-" * 80)
    
    # Test with first example
    test_text = EXAMPLE_TEXTS[0]
    print(f"\nTest Text: '{test_text}'")
    print("\nPredictions from all models:")
    
    all_results = {}
    for model_name, pipe in loaded_models.items():
        result = predict_sentiment(test_text, pipe)
        all_results[model_name] = result
        print(f"  {model_name:15s}: {result['label']:10s} (confidence: {result['score']:.4f})")
    
    print("\n[3] BATCH PREDICTION EXAMPLE")
    print("-" * 80)
    
    # Batch prediction with best model (if available)
    if 'RoBERTa' in loaded_models:
        print(f"\nBatch predicting with RoBERTa on {len(EXAMPLE_TEXTS)} texts...")
        
        predictions = predict_batch(EXAMPLE_TEXTS, loaded_models['RoBERTa'])
        
        print("\nResults:")
        for pred in predictions:
            status = "✓" if pred['label'] != 'ERROR' else "✗"
            print(f"{status} '{pred['text'][:40]:40s}' → {pred.get('label', 'ERROR'):10s} ({pred.get('score', 0):.4f})")
    
    print("\n[4] EXPORT PREDICTIONS")
    print("-" * 80)
    
    # Save example predictions
    output_data = []
    for model_name, pipe in loaded_models.items():
        for text in EXAMPLE_TEXTS:
            result = predict_sentiment(text, pipe)
            output_data.append({
                'model': model_name,
                'text': result['text'],
                'prediction': result['label'],
                'confidence': result['score']
            })
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv('example_predictions.csv', index=False)
    print("✓ Saved predictions to: example_predictions.csv")

else:
    print("\n⚠ No models loaded. Make sure to run train_transformer_models.py first!")

print("\n[5] USAGE INSTRUCTIONS")
print("-" * 80)
print("""
To use a model for prediction:

    from transformers import pipeline
    pipe = pipeline("text-classification", model="./sentiment_models/RoBERTa")
    
    result = pipe("Your text here")
    print(result)  # [{'label': 'positive', 'score': 0.9999}]

For batch predictions:

    texts = ["Text 1", "Text 2", "Text 3"]
    results = pipe(texts)
    for result in results:
        print(result)

Recommended models:
  - RoBERTa: Best overall performance
  - DistilBERT: Fast inference with good performance
  - BERT: Balanced performance
  - ALBERT: Memory efficient
  - XLNET: Complex context understanding
""")

print("\n" + "=" * 80)
print("✓ INFERENCE SCRIPT READY")
print("=" * 80)
