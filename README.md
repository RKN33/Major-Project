# Fraud Detection in Financial Transactions

**Name:** Rishit Kumar Nayak
**Roll No:** AI ML-A6/March-9199
**Institute:** InternsElite

---

## Project Structure

```
fraud_detection/
├── data/
│   └── transactions.csv
├── models/
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── meta_model.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
├── outputs/
│   ├── eda_overview.png
│   ├── correlation_matrix.png
│   ├── model_evaluation.png
│   ├── feature_importance.png
│   ├── shap_importance.png
│   ├── shap_beeswarm.png
│   ├── score_distribution.png
│   └── model_results.csv
└── src/
    ├── generate_data.py
    ├── eda.py
    ├── feature_engineering.py
    ├── train.py
    └── predict.py
```

## Installation

```bash
pip install xgboost lightgbm scikit-learn imbalanced-learn shap pandas numpy matplotlib seaborn joblib
```

## How to Run

```bash
python src/generate_data.py
python src/eda.py
python src/train.py
python src/predict.py
python src/predict.py --batch data/transactions.csv
```

## Results

| Model    | AUPRC  | AUROC  | F1     |
|----------|--------|--------|--------|
| XGBoost  | 0.8294 | 0.9856 | 0.6998 |
| LightGBM | 0.8276 | 0.9852 | 0.6819 |
| **Ensemble** | **0.8322** | **0.9861** | **0.7714** |

## Approach

- 50,000 synthetic transactions with realistic fraud patterns (3.5% fraud rate)
- 28 engineered features: temporal, velocity, rolling aggregates, behavioral flags
- SMOTE for class imbalance handling
- Stacked ensemble: XGBoost + LightGBM base learners → Logistic Regression meta-learner
- SHAP values for per-transaction explainability
- Time-based train/test split to prevent data leakage
