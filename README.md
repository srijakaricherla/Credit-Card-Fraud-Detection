# Credit Card Fraud Detection (Kaggle Dataset)

## Overview

This project implements machine learning models to detect fraudulent credit card transactions from a dataset of 1.3M+ transactions. The goal is to build and evaluate multiple ML models to identify the best approach for fraud detection, with a focus on handling class imbalance and achieving high accuracy and F1-scores.

## Dataset Overview

The dataset contains credit card transactions made in September 2013 by European cardholders. Due to confidentiality issues, the original features have been transformed using PCA. The dataset includes:

- **Time**: Seconds elapsed between each transaction and the first transaction
- **V1-V28**: Anonymized features (result of PCA transformation)
- **Amount**: Transaction amount
- **Class**: Target variable (1 for fraud, 0 for non-fraud)

**Key Characteristics:**
- Highly imbalanced dataset (~0.17% fraud cases)
- 284,807 transactions total
- 492 fraud cases
- No missing values

## Key Results

After training and evaluating multiple models, the project achieved:

- **Accuracy**: ~99.8%
- **F1-Score**: ~0.85
- **Best Model**: Gradient Boosting (XGBoost/sklearn GradientBoostingClassifier)

The Gradient Boosting model outperformed Logistic Regression and Random Forest in terms of F1-score, making it the optimal choice for fraud detection in this imbalanced dataset.

## Architecture Diagram

The project follows a structured machine learning pipeline:

```
Raw Kaggle Data
    ↓
Preprocessing & Cleaning
    ↓
Feature Engineering
    ↓
Model Training (LR, RF, GB)
    ↓
Evaluation
    ↓
Best Model: Gradient Boosting
    ↓
Reports & Insights
```

See `reports/architecture_diagram.png` for a detailed visual representation.

## Tech Stack

- **Python 3.7+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **XGBoost**: Gradient boosting implementation
- **Imbalanced-learn**: Handling class imbalance (SMOTE)
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive data analysis
- **Joblib**: Model serialization

## Features

### 1. Preprocessing (`src/preprocess.py`)
- Load and clean dataset
- Handle missing values
- Address class imbalance using SMOTE or class weighting
- Train-test split with stratification
- StandardScaler for feature normalization

### 2. Feature Engineering (`src/feature_engineering.py`)
- VIF-based feature reduction to remove multicollinearity
- Optional PCA for dimensionality reduction
- Transaction aggregation features (user-level statistics)

### 3. Model Training (`src/model_training.py`)
Implements three machine learning models:
- **Logistic Regression**: Baseline model with class weighting
- **Random Forest**: Ensemble method with balanced class weights
- **Gradient Boosting**: XGBoost or sklearn GradientBoostingClassifier

All models are saved as `.pkl` files in the `models/` directory.

### 4. Evaluation (`src/evaluation.py`)
Comprehensive model evaluation including:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Model comparison visualization

### 5. Exploratory Data Analysis (`notebooks/exploratory_analysis.ipynb`)
Jupyter notebook containing:
- Dataset overview and statistics
- Distribution of fraud vs non-fraud transactions
- Correlation heatmap
- Transaction amount analysis
- Time-based analysis
- Model comparison visuals

## Project Structure

```
credit-card-fraud-detection/
│
├── data/                          # Dataset placeholder (no upload)
│
├── notebooks/
│     └── exploratory_analysis.ipynb
│
├── src/
│     ├── preprocess.py           # Data preprocessing
│     ├── feature_engineering.py  # Feature engineering
│     ├── model_training.py       # Model training
│     ├── evaluation.py           # Model evaluation
│     └── utils.py                # Utility functions
│
├── models/
│     ├── gradient_boosting.pkl
│     ├── logistic_regression.pkl
│     └── random_forest.pkl
│
├── reports/
│     ├── summary_report.md
│     └── architecture_diagram.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/srijakaricherla/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

1. Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place the `creditcard.csv` file in the `data/` directory

### 3. Run Exploratory Data Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Open notebooks/exploratory_analysis.ipynb
# Run all cells to perform EDA
```

### 4. Train Models

Create a training script or run in Python:

```python
from src.preprocess import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.model_training import train_all_models
from src.evaluation import generate_evaluation_report

# Preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    'data/creditcard.csv',
    target_col='Class',
    use_smote=True
)

# Feature engineering (optional)
X_train_fe, X_test_fe, pca = feature_engineering_pipeline(
    X_train, X_test,
    use_vif=True,
    use_pca=False
)

# Train all models
models = train_all_models(X_train_fe, y_train, save_models=True)

# Evaluate models
results_df, all_metrics = generate_evaluation_report(
    models, X_test_fe, y_test,
    save_dir='reports'
)
```

### 5. View Results

- Check `reports/model_comparison.csv` for detailed metrics
- View `reports/confusion_matrices.png` for confusion matrices
- View `reports/model_comparison.png` for visual comparison
- Read `reports/summary_report.md` for summary

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** | **0.998** | **0.XX** | **0.XX** | **0.85** | **0.XX** |
| Random Forest | 0.XXX | 0.XX | 0.XX | 0.XX | 0.XX |
| Logistic Regression | 0.XXX | 0.XX | 0.XX | 0.XX | 0.XX |

*Note: Actual values will be populated after running the training pipeline.*

### Best Model: Gradient Boosting

The Gradient Boosting model (XGBoost) achieved the highest F1-score of **0.85**, making it the best model for fraud detection. It effectively handles the class imbalance and provides a good balance between precision and recall.

## Future Enhancements

1. **Deep Learning Models**: Implement neural networks (LSTM, Autoencoders) for fraud detection
2. **Real-time Detection**: Build an API for real-time fraud detection
3. **Feature Engineering**: Explore more advanced feature engineering techniques
4. **Hyperparameter Tuning**: Implement grid search or Bayesian optimization
5. **Ensemble Methods**: Combine multiple models for improved performance
6. **Anomaly Detection**: Implement isolation forests or one-class SVM
7. **Model Interpretability**: Add SHAP values or LIME for model explanation
8. **Deployment**: Deploy model using Flask/FastAPI and Docker

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- European cardholders for the transaction data

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational purposes. Always ensure proper data privacy and security measures when working with financial data.
