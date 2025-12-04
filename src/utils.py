"""
Utility functions for Credit Card Fraud Detection project.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)


def save_model(model, filename):
    """Save trained model to models directory."""
    models_dir = get_project_root() / 'models'
    models_dir.mkdir(exist_ok=True)
    filepath = models_dir / filename
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename):
    """Load trained model from models directory."""
    models_dir = get_project_root() / 'models'
    filepath = models_dir / filename
    return joblib.load(filepath)


def save_results(results, filename):
    """Save results to reports directory."""
    reports_dir = get_project_root() / 'reports'
    reports_dir.mkdir(exist_ok=True)
    filepath = reports_dir / filename
    results.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

