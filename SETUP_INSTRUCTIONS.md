# Setup Instructions

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   - Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the `data/` directory

3. **Generate Architecture Diagram** (Optional)
   ```bash
   python generate_architecture_diagram.py
   ```
   This creates `reports/architecture_diagram.png`

4. **Run the Pipeline**
   ```bash
   python main.py
   ```
   This will:
   - Preprocess the data
   - Engineer features
   - Train all models (Logistic Regression, Random Forest, Gradient Boosting)
   - Evaluate and compare models
   - Save results to `reports/` directory

5. **Explore Data** (Optional)
   ```bash
   jupyter notebook
   ```
   Open `notebooks/exploratory_analysis.ipynb` for EDA

## Git Operations

Since git is not currently available in your environment, you can run these commands manually when git is installed:

```bash
git add .
git commit -m "Initial complete Credit Card Fraud Detection ML project"
git push
```

## Project Structure

```
credit-card-fraud-detection/
├── data/                    # Place your dataset here
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code modules
├── models/                 # Trained models (generated)
├── reports/                # Evaluation reports and diagrams
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Notes

- Model files (`.pkl`) will be generated in `models/` after training
- Reports will be saved to `reports/` after evaluation
- The architecture diagram script requires matplotlib to be installed

