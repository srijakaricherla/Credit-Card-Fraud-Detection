"""
Main script to run the complete Credit Card Fraud Detection pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocess import preprocess_pipeline
from feature_engineering import feature_engineering_pipeline
from model_training import train_all_models
from evaluation import generate_evaluation_report


def main():
    """Run the complete ML pipeline."""
    print("="*70)
    print("CREDIT CARD FRAUD DETECTION - ML PIPELINE")
    print("="*70)
    
    # Configuration
    data_path = 'data/creditcard.csv'
    target_col = 'Class'
    use_smote = True
    test_size = 0.2
    
    # Step 1: Preprocessing
    print("\n[STEP 1] Preprocessing...")
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
            data_path,
            target_col=target_col,
            use_smote=use_smote,
            test_size=test_size
        )
        print("✓ Preprocessing complete")
    except FileNotFoundError:
        print(f"✗ Error: Dataset not found at {data_path}")
        print("Please place your dataset in the data/ directory")
        return
    except Exception as e:
        print(f"✗ Error during preprocessing: {e}")
        return
    
    # Step 2: Feature Engineering
    print("\n[STEP 2] Feature Engineering...")
    try:
        X_train_fe, X_test_fe, pca = feature_engineering_pipeline(
            X_train, X_test,
            use_vif=True,
            use_pca=False,  # Set to True if you want to use PCA
            vif_threshold=10
        )
        print("✓ Feature engineering complete")
    except Exception as e:
        print(f"✗ Error during feature engineering: {e}")
        return
    
    # Step 3: Model Training
    print("\n[STEP 3] Training Models...")
    try:
        models = train_all_models(X_train_fe, y_train, save_models=True)
        print("✓ Model training complete")
    except Exception as e:
        print(f"✗ Error during model training: {e}")
        return
    
    # Step 4: Evaluation
    print("\n[STEP 4] Evaluating Models...")
    try:
        results_df, all_metrics = generate_evaluation_report(
            models, X_test_fe, y_test,
            save_dir='reports'
        )
        print("✓ Evaluation complete")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - models/*.pkl (trained models)")
    print("  - reports/model_comparison.csv")
    print("  - reports/confusion_matrices.png")
    print("  - reports/model_comparison.png")
    print("\nBest Model:", results_df.iloc[0]['Model'])
    print("F1-Score:", f"{results_df.iloc[0]['F1-Score']:.4f}")
    print("Accuracy:", f"{results_df.iloc[0]['Accuracy']:.4f}")


if __name__ == '__main__':
    main()

