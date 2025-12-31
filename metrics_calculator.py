"""
Advanced Metrics Calculator
Comprehensive metrics calculation including ROC/AUC, PR curves, confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, hamming_loss
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class MetricsCalculator:
    """Calculate comprehensive metrics for model evaluation"""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Initialize metrics calculator
        
        Args:
            y_true: True labels (0, 1 or 'negative', 'positive')
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            model_name: Name of the model
        """
        self.y_true = self._encode_labels(y_true)
        self.y_pred = self._encode_labels(y_pred)
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have same length")
    
    def _encode_labels(self, y):
        """Encode string labels to numeric"""
        if isinstance(y, list):
            y = np.array(y)
        
        if len(y) > 0 and isinstance(y[0], str):
            unique_labels = np.unique(y)
            mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            return np.array([mapping[label] for label in y])
        
        return np.array(y)
    
    def calculate_all(self):
        """Calculate all metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred).tolist()
        
        # ROC/AUC (for binary classification)
        if len(np.unique(self.y_true)) == 2:
            if self.y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            else:
                metrics['roc_auc'] = None
        
        # Classification report
        report = classification_report(
            self.y_true, self.y_pred,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Per-class metrics
        for class_idx in np.unique(self.y_true):
            class_mask = self.y_true == class_idx
            class_tp = np.sum((self.y_pred == class_idx) & class_mask)
            class_fp = np.sum((self.y_pred == class_idx) & ~class_mask)
            class_fn = np.sum((self.y_pred != class_idx) & class_mask)
            class_tn = np.sum((self.y_pred != class_idx) & ~class_mask)
            
            metrics[f'class_{class_idx}_tp'] = int(class_tp)
            metrics[f'class_{class_idx}_fp'] = int(class_fp)
            metrics[f'class_{class_idx}_fn'] = int(class_fn)
            metrics[f'class_{class_idx}_tn'] = int(class_tn)
        
        return metrics
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, figsize=(8, 6), save_path=None):
        """Plot ROC curve"""
        if self.y_pred_proba is None or len(np.unique(self.y_true)) != 2:
            print("ROC curve requires binary classification and probabilities")
            return None
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_pr_curve(self, figsize=(8, 6), save_path=None):
        """Plot Precision-Recall curve"""
        if self.y_pred_proba is None or len(np.unique(self.y_true)) != 2:
            print("PR curve requires binary classification and probabilities")
            return None
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="best")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_summary(self):
        """Get summary of metrics as string"""
        metrics = self.calculate_all()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  Model Evaluation Report: {self.model_name}
╚══════════════════════════════════════════════════════════════╝

📊 OVERALL METRICS
─────────────────────────────────────────────────────────────
Accuracy:   {metrics['accuracy']:.4f}
Precision:  {metrics['precision']:.4f}
Recall:     {metrics['recall']:.4f}
F1-Score:   {metrics['f1']:.4f}

📈 CONFUSION MATRIX
─────────────────────────────────────────────────────────────
{np.array(metrics['confusion_matrix'])}

🎯 PER-CLASS METRICS
─────────────────────────────────────────────────────────────
"""
        
        if 'classification_report' in metrics:
            for class_name, class_metrics in metrics['classification_report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    summary += f"\nClass {class_name}:\n"
                    for metric_name, value in class_metrics.items():
                        if isinstance(value, (int, float)):
                            summary += f"  {metric_name}: {value:.4f}\n"
        
        return summary
    
    def export_to_json(self, file_path):
        """Export metrics to JSON"""
        metrics = self.calculate_all()
        
        # Convert numpy arrays to lists
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        metrics = convert_types(metrics)
        metrics['model_name'] = self.model_name
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics exported to {file_path}")


def calculate_model_metrics(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Quick function to calculate all metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        model_name: Model name
        
    Returns:
        Dictionary with all metrics
    """
    calculator = MetricsCalculator(y_true, y_pred, y_pred_proba, model_name)
    return calculator.calculate_all()


def generate_metrics_report(y_true, y_pred, y_pred_proba=None, model_name="Model", save_dir="./metrics"):
    """
    Generate comprehensive metrics report with visualizations
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        model_name: Model name
        save_dir: Directory to save figures
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    calculator = MetricsCalculator(y_true, y_pred, y_pred_proba, model_name)
    
    # Print summary
    print(calculator.get_summary())
    
    # Save metrics as JSON
    json_path = f"{save_dir}/{model_name}_metrics.json"
    calculator.export_to_json(json_path)
    
    # Save plots
    try:
        cm_fig = calculator.plot_confusion_matrix(
            save_path=f"{save_dir}/{model_name}_confusion_matrix.png"
        )
        print(f"✅ Saved: {save_dir}/{model_name}_confusion_matrix.png")
    except:
        pass
    
    try:
        roc_fig = calculator.plot_roc_curve(
            save_path=f"{save_dir}/{model_name}_roc_curve.png"
        )
        if roc_fig:
            print(f"✅ Saved: {save_dir}/{model_name}_roc_curve.png")
    except:
        pass
    
    try:
        pr_fig = calculator.plot_pr_curve(
            save_path=f"{save_dir}/{model_name}_pr_curve.png"
        )
        if pr_fig:
            print(f"✅ Saved: {save_dir}/{model_name}_pr_curve.png")
    except:
        pass


# Example usage
if __name__ == "__main__":
    # Sample data
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    y_proba = [0.1, 0.9, 0.85, 0.2, 0.95, 0.4, 0.15, 0.1, 0.9, 0.2]
    
    # Calculate metrics
    metrics = calculate_model_metrics(y_true, y_pred, y_proba, "TestModel")
    
    print("Metrics calculated successfully!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Generate full report
    print("\nGenerating detailed report...")
    generate_metrics_report(y_true, y_pred, y_proba, "TestModel", "./test_metrics")
