"""
Model training module for Credit Card Fraud Detection.
Trains Logistic Regression, Random Forest, and Gradient Boosting models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, y_train, class_weight='balanced', random_state=42):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    class_weight : str or dict
        Class weight strategy
    random_state : int
        Random seed
        
    Returns:
    --------
    model : LogisticRegression
        Trained model
    """
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    print("Logistic Regression training complete!")
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, class_weight='balanced', 
                       random_state=42, n_jobs=-1):
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_estimators : int
        Number of trees
    class_weight : str or dict
        Class weight strategy
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    print("Random Forest training complete!")
    
    return model


def train_gradient_boosting(X_train, y_train, use_xgboost=True, n_estimators=100, 
                           learning_rate=0.1, random_state=42):
    """
    Train Gradient Boosting model (XGBoost or sklearn).
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    use_xgboost : bool
        Whether to use XGBoost (True) or sklearn GradientBoosting (False)
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Learning rate
    random_state : int
        Random seed
        
    Returns:
    --------
    model : XGBClassifier or GradientBoostingClassifier
        Trained model
    """
    print("\n" + "="*50)
    print("Training Gradient Boosting...")
    print("="*50)
    
    if use_xgboost:
        try:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            )
            print("Using XGBoost")
        except ImportError:
            print("XGBoost not available, using sklearn GradientBoosting")
            use_xgboost = False
    
    if not use_xgboost:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            max_depth=5
        )
        print("Using sklearn GradientBoosting")
    
    model.fit(X_train, y_train)
    print("Gradient Boosting training complete!")
    
    return model


def train_all_models(X_train, y_train, save_models=True):
    """
    Train all models and save them.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    save_models : bool
        Whether to save models to disk
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    """
    models = {}
    
    # Train Logistic Regression
    models['logistic_regression'] = train_logistic_regression(X_train, y_train)
    
    # Train Random Forest
    models['random_forest'] = train_random_forest(X_train, y_train)
    
    # Train Gradient Boosting
    models['gradient_boosting'] = train_gradient_boosting(X_train, y_train)
    
    # Save models
    if save_models:
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            filepath = models_dir / f"{name}.pkl"
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
    
    return models

