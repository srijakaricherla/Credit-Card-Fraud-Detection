"""
Model evaluation module for Credit Card Fraud Detection.
Computes accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model and return metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str
        Name of the model
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['Confusion_Matrix'] = cm
    
    return metrics


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and create comparison table.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Comparison table of all models
    all_metrics : dict
        Dictionary of all metrics for each model
    """
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70)
    
    all_metrics = {}
    results_list = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        all_metrics[name] = metrics
        
        # Add to results list (excluding confusion matrix)
        result_row = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
        results_list.append(result_row)
        
        # Print metrics
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
        if metrics['ROC-AUC'] is not None:
            print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Identify best model
    best_model = results_df.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    
    return results_df, all_metrics


def plot_confusion_matrices(all_metrics, save_path=None):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    -----------
    all_metrics : dict
        Dictionary of all metrics
    save_path : str or Path
        Path to save the plot
    """
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, metrics) in enumerate(all_metrics.items()):
        cm = metrics['Confusion_Matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot bar chart comparing model metrics.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results comparison table
    save_path : str or Path
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def generate_evaluation_report(models, X_test, y_test, save_dir=None):
    """
    Generate complete evaluation report.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    save_dir : str or Path
        Directory to save reports
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Comparison table
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Evaluate all models
    results_df, all_metrics = evaluate_all_models(models, X_test, y_test)
    
    # Save results to CSV
    if save_dir:
        results_df.to_csv(save_dir / 'model_comparison.csv', index=False)
    
    # Plot confusion matrices
    if save_dir:
        plot_confusion_matrices(all_metrics, save_path=save_dir / 'confusion_matrices.png')
    
    # Plot comparison
    if save_dir:
        plot_model_comparison(results_df, save_path=save_dir / 'model_comparison.png')
    
    return results_df, all_metrics

