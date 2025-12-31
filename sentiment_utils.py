"""
Sentiment Analysis Utility Functions
Helper functions untuk training, inference, dan evaluation
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple

class SentimentAnalyzer:
    """Main class untuk sentiment analysis dengan transformer models"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize sentiment analyzer
        
        Args:
            model_path: Path ke trained model
            use_gpu: Whether to use GPU if available
        """
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.model_path = model_path
        self.pipe = None
        self.load_model()
    
    def load_model(self):
        """Load model from path"""
        try:
            self.pipe = pipeline(
                "text-classification",
                model=self.model_path,
                device=0 if self.device == "cuda" else -1
            )
            print(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            self.pipe = None
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.pipe:
            return {'error': 'Model not loaded'}
        
        try:
            result = self.pipe(text, truncation=True)[0]
            return {
                'text': text,
                'sentiment': result['label'].lower(),
                'confidence': result['score']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of predictions
        """
        if not self.pipe:
            return [{'error': 'Model not loaded'} for _ in texts]
        
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                results = self.pipe(batch, truncation=True)
                for text, result in zip(batch, results):
                    predictions.append({
                        'text': text,
                        'sentiment': result['label'].lower(),
                        'confidence': result['score']
                    })
            except Exception as e:
                for text in batch:
                    predictions.append({'text': text, 'error': str(e)})
        
        return predictions


def compare_models(test_texts: List[str], test_labels: List[str], model_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Compare multiple models on test set
    
    Args:
        test_texts: List of test texts
        test_labels: List of true labels
        model_paths: Dictionary of model_name -> model_path
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating {model_name}...")
        analyzer = SentimentAnalyzer(model_path)
        
        predictions = analyzer.predict_batch(test_texts)
        predicted_labels = [p.get('sentiment', 'error') for p in predictions]
        
        # Filter out errors
        valid_indices = [i for i, p in enumerate(predictions) if 'error' not in p]
        
        if valid_indices:
            valid_labels = [test_labels[i] for i in valid_indices]
            valid_predictions = [predicted_labels[i] for i in valid_indices]
            
            accuracy = accuracy_score(valid_labels, valid_predictions)
            precision = precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
            recall = recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
            f1 = f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Valid Samples': len(valid_indices)
            })
        else:
            print(f"  ✗ No valid predictions for {model_name}")
    
    return pd.DataFrame(results).sort_values('F1-Score', ascending=False)


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = 'model_comparison.png'):
    """
    Plot model comparison results
    
    Args:
        comparison_df: DataFrame from compare_models()
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Transformer Models Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = comparison_df.sort_values(metric, ascending=False)
        
        bars = ax.barh(data['Model'], data[metric])
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xlim([0, 1])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {save_path}")
    plt.close()


def evaluate_model(model_path: str, test_texts: List[str], test_labels: List[str]) -> Dict:
    """
    Evaluate a single model
    
    Args:
        model_path: Path to model
        test_texts: List of test texts
        test_labels: List of true labels
        
    Returns:
        Dictionary with evaluation metrics and confusion matrix
    """
    analyzer = SentimentAnalyzer(model_path)
    predictions = analyzer.predict_batch(test_texts)
    predicted_labels = [p.get('sentiment', 'error') for p in predictions]
    
    # Filter out errors
    valid_indices = [i for i, p in enumerate(predictions) if 'error' not in p]
    valid_labels = [test_labels[i] for i in valid_indices]
    valid_predictions = [predicted_labels[i] for i in valid_indices]
    
    results = {
        'accuracy': accuracy_score(valid_labels, valid_predictions),
        'precision': precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
        'recall': recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
        'f1': f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(valid_labels, valid_predictions),
        'valid_samples': len(valid_indices),
        'total_samples': len(test_texts)
    }
    
    return results


def batch_predict_to_csv(model_path: str, input_csv: str, output_csv: str, text_column: str = 'text'):
    """
    Predict sentiment for CSV file and save results
    
    Args:
        model_path: Path to model
        input_csv: Input CSV file
        output_csv: Output CSV file
        text_column: Name of text column in CSV
    """
    df = pd.read_csv(input_csv)
    analyzer = SentimentAnalyzer(model_path)
    
    predictions = analyzer.predict_batch(df[text_column].tolist())
    
    df['predicted_sentiment'] = [p.get('sentiment', 'error') for p in predictions]
    df['confidence'] = [p.get('confidence', 0) for p in predictions]
    
    df.to_csv(output_csv, index=False)
    print(f"✓ Predictions saved to {output_csv}")


def ensemble_predict(test_text: str, model_paths: Dict[str, str], voting: str = 'majority') -> Dict:
    """
    Ensemble prediction from multiple models
    
    Args:
        test_text: Input text
        model_paths: Dictionary of model_name -> model_path
        voting: 'majority' or 'confidence' based voting
        
    Returns:
        Ensemble prediction result
    """
    predictions = {}
    
    for model_name, model_path in model_paths.items():
        analyzer = SentimentAnalyzer(model_path)
        result = analyzer.predict(test_text)
        predictions[model_name] = result
    
    if voting == 'majority':
        sentiments = [p.get('sentiment') for p in predictions.values() if 'error' not in p]
        if sentiments:
            final_sentiment = max(set(sentiments), key=sentiments.count)
        else:
            final_sentiment = 'unknown'
    
    elif voting == 'confidence':
        valid_preds = [p for p in predictions.values() if 'error' not in p]
        if valid_preds:
            final_pred = max(valid_preds, key=lambda x: x.get('confidence', 0))
            final_sentiment = final_pred.get('sentiment', 'unknown')
        else:
            final_sentiment = 'unknown'
    
    return {
        'text': test_text,
        'ensemble_prediction': final_sentiment,
        'individual_predictions': predictions,
        'voting_method': voting
    }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("SENTIMENT ANALYSIS UTILITY FUNCTIONS")
    print("=" * 80)
    
    # Example 1: Single prediction
    print("\n[1] SINGLE PREDICTION")
    print("-" * 80)
    
    analyzer = SentimentAnalyzer('./sentiment_models/RoBERTa')
    result = analyzer.predict("I love this product!")
    print(f"Text: {result.get('text')}")
    print(f"Sentiment: {result.get('sentiment')}")
    print(f"Confidence: {result.get('confidence'):.4f}")
    
    # Example 2: Batch prediction
    print("\n[2] BATCH PREDICTION")
    print("-" * 80)
    
    texts = [
        "This is great!",
        "I hate this.",
        "It's okay.",
    ]
    
    results = analyzer.predict_batch(texts)
    for result in results:
        if 'error' not in result:
            print(f"'{result['text'][:30]:30s}' → {result['sentiment']:10s} ({result['confidence']:.4f})")
    
    # Example 3: Ensemble prediction
    print("\n[3] ENSEMBLE PREDICTION")
    print("-" * 80)
    
    model_paths = {
        'RoBERTa': './sentiment_models/RoBERTa',
        'BERT': './sentiment_models/BERT',
    }
    
    text = "This product is fantastic and I recommend it!"
    ensemble_result = ensemble_predict(text, model_paths, voting='confidence')
    print(f"Text: {ensemble_result['text']}")
    print(f"Ensemble Prediction: {ensemble_result['ensemble_prediction']}")
    print(f"Voting Method: {ensemble_result['voting_method']}")
    
    print("\n" + "=" * 80)
    print("✓ UTILITIES READY TO USE")
    print("=" * 80)
