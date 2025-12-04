"""
Data preprocessing module for Credit Card Fraud Detection.
Handles data loading, cleaning, handling class imbalance, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath):
    """
    Load dataset and perform basic cleaning.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Cleaned dataframe
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum().sum()}")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    print(f"Cleaned dataset shape: {df.shape}")
    return df


def handle_class_imbalance(X, y, method='smote', sampling_strategy=0.1):
    """
    Handle class imbalance using SMOTE or class weighting.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    method : str
        'smote' for SMOTE oversampling, 'undersample' for random undersampling
    sampling_strategy : float
        Ratio for SMOTE (0.1 means 10% of majority class)
        
    Returns:
    --------
    X_resampled : array-like
        Resampled feature matrix
    y_resampled : array-like
        Resampled target vector
    """
    print(f"\nOriginal class distribution:")
    print(f"Class 0 (Non-fraud): {np.sum(y == 0)}")
    print(f"Class 1 (Fraud): {np.sum(y == 1)}")
    print(f"Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}:1")
    
    if method == 'smote':
        print("\nApplying SMOTE oversampling...")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        print("\nApplying random undersampling...")
        undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
    else:
        print("\nNo resampling applied (using class weights instead)")
        return X, y
    
    print(f"\nResampled class distribution:")
    print(f"Class 0 (Non-fraud): {np.sum(y_resampled == 0)}")
    print(f"Class 1 (Fraud): {np.sum(y_resampled == 1)}")
    print(f"New ratio: {np.sum(y_resampled == 0) / np.sum(y_resampled == 1):.2f}:1")
    
    return X_resampled, y_resampled


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Train-test split
    """
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
        
    Returns:
    --------
    X_train_scaled : array-like
        Scaled training features
    X_test_scaled : array-like
        Scaled test features
    scaler : StandardScaler
        Fitted scaler object
    """
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath, target_col='Class', use_smote=True, test_size=0.2):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to dataset
    target_col : str
        Name of target column
    use_smote : bool
        Whether to use SMOTE
    test_size : float
        Test set proportion
        
    Returns:
    --------
    Preprocessed data and scaler
    """
    # Load and clean
    df = load_and_clean_data(filepath)
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle class imbalance
    if use_smote:
        X, y = handle_class_imbalance(X, y, method='smote')
    
    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

