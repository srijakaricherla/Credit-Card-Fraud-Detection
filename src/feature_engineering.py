"""
Feature engineering module for Credit Card Fraud Detection.
Includes VIF-based feature reduction, PCA, and transaction aggregation features.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


def calculate_vif(X, threshold=10):
    """
    Calculate Variance Inflation Factor for features and remove high VIF features.
    
    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        Feature matrix
    threshold : float
        VIF threshold above which features are removed
        
    Returns:
    --------
    X_reduced : array-like
        Feature matrix with high VIF features removed
    removed_features : list
        List of removed feature indices/names
    """
    if isinstance(X, pd.DataFrame):
        X_df = X
    else:
        X_df = pd.DataFrame(X)
    
    print(f"\nCalculating VIF for {X_df.shape[1]} features...")
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns if isinstance(X, pd.DataFrame) else range(X_df.shape[1])
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) 
                       for i in range(X_df.shape[1])]
    
    # Remove features with VIF > threshold
    high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
    
    if len(high_vif_features) > 0:
        print(f"Removing {len(high_vif_features)} features with VIF > {threshold}")
        if isinstance(X, pd.DataFrame):
            X_reduced = X.drop(columns=high_vif_features)
        else:
            # For numpy arrays, we need to keep track of indices
            feature_indices = [i for i, feat in enumerate(vif_data["Feature"]) 
                             if feat not in high_vif_features]
            X_reduced = X[:, feature_indices]
        removed_features = high_vif_features
    else:
        print("No features with high VIF found")
        X_reduced = X
        removed_features = []
    
    print(f"Reduced feature count: {X_df.shape[1]} -> {X_reduced.shape[1] if hasattr(X_reduced, 'shape') else len(X_reduced[0])}")
    
    return X_reduced, removed_features


def apply_pca(X, n_components=None, variance_threshold=0.95):
    """
    Apply Principal Component Analysis for dimensionality reduction.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    n_components : int or None
        Number of components (if None, use variance_threshold)
    variance_threshold : float
        Cumulative variance threshold to determine n_components
        
    Returns:
    --------
    X_pca : array-like
        Transformed feature matrix
    pca : PCA
        Fitted PCA object
    """
    print(f"\nApplying PCA...")
    
    if n_components is None:
        # Find number of components that explain variance_threshold of variance
        pca_temp = PCA()
        pca_temp.fit(X)
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        print(f"Selected {n_components} components to explain {variance_threshold*100}% variance")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA transformation complete: {X.shape[1]} -> {X_pca.shape[1]} features")
    print(f"Explained variance: {explained_variance:.2%}")
    
    return X_pca, pca


def create_aggregation_features(df, group_by_col='Time', target_col='Class'):
    """
    Create user-level or time-based aggregation features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    group_by_col : str
        Column to group by (e.g., 'Time' for time-based, user ID for user-based)
    target_col : str
        Target column name
        
    Returns:
    --------
    df_with_agg : pandas.DataFrame
        Dataframe with aggregation features added
    """
    print(f"\nCreating aggregation features grouped by '{group_by_col}'...")
    
    df_with_agg = df.copy()
    
    # Numeric columns to aggregate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if group_by_col in numeric_cols:
        numeric_cols.remove(group_by_col)
    
    # Create aggregation features
    if group_by_col in df.columns:
        agg_features = df.groupby(group_by_col)[numeric_cols].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Flatten column names
        agg_features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                               for col in agg_features.columns]
        
        # Merge back
        df_with_agg = df_with_agg.merge(agg_features, on=group_by_col, how='left')
        print(f"Added {len(agg_features.columns) - 1} aggregation features")
    else:
        print(f"Warning: '{group_by_col}' not found in dataframe")
    
    return df_with_agg


def feature_engineering_pipeline(X_train, X_test, use_vif=True, use_pca=False, 
                                  vif_threshold=10, pca_variance=0.95):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    use_vif : bool
        Whether to apply VIF-based feature reduction
    use_pca : bool
        Whether to apply PCA
    vif_threshold : float
        VIF threshold
    pca_variance : float
        PCA variance threshold
        
    Returns:
    --------
    Processed features and transformers
    """
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # VIF-based feature reduction
    if use_vif:
        X_train_processed, removed_features = calculate_vif(
            X_train_processed, threshold=vif_threshold
        )
        if len(removed_features) > 0 and isinstance(X_test, pd.DataFrame):
            X_test_processed = X_test_processed.drop(columns=removed_features)
        elif len(removed_features) > 0:
            # Handle numpy array case
            feature_indices = [i for i in range(X_test.shape[1]) 
                             if i not in [int(f) for f in removed_features if str(f).isdigit()]]
            X_test_processed = X_test[:, feature_indices]
    
    # PCA
    if use_pca:
        X_train_processed, pca = apply_pca(X_train_processed, variance_threshold=pca_variance)
        X_test_processed = pca.transform(X_test_processed)
        return X_train_processed, X_test_processed, pca
    
    return X_train_processed, X_test_processed, None

