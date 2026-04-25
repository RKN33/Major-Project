import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from feature_engineering import engineer_features, FEATURE_COLS

SEED = 42
OUTDIR = '/home/claude/fraud_detection/outputs'
MDIR = '/home/claude/fraud_detection/models'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MDIR, exist_ok=True)

raw = pd.read_csv('/home/claude/fraud_detection/data/transactions.csv')
print(f"Loaded: {len(raw):,} rows | Fraud: {raw['isFraud'].sum():,} ({raw['isFraud'].mean()*100:.2f}%)")

df = engineer_features(raw)
available = [c for c in FEATURE_COLS if c in df.columns]
X = df[available].fillna(0)
y = df['isFraud']

split = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

sm = SMOTE(random_state=SEED, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — Fraud: {y_res.sum():,} | Legit: {(y_res==0).sum():,}")

imbalance_ratio = int((y_train == 0).sum() / (y_train == 1).sum())

xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=imbalance_ratio,
    eval_metric='aucpr', use_label_encoder=False,
    random_state=SEED, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=False)

lgb_model = lgb.LGBMClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=imbalance_ratio,
    random_state=SEED, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_res, y_res,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

xgb_oof = np.zeros(len(X_train))
lgb_oof = np.zeros(len(X_train))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr = y_train.iloc[tr_idx]

    _xgb = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                               scale_pos_weight=imbalance_ratio, use_label_encoder=False,
                               random_state=SEED, verbosity=0, n_jobs=-1)
    _xgb.fit(Xtr, ytr, verbose=False)
    xgb_oof[val_idx] = _xgb.predict_proba(Xval)[:, 1]

    _lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                scale_pos_weight=imbalance_ratio,
                                random_state=SEED, verbose=-1, n_jobs=-1)
    _lgb.fit(Xtr, ytr)
    lgb_oof[val_idx] = _lgb.predict_proba(Xval)[:, 1]
    print(f"  Fold {fold+1}/5 complete")

meta_train = np.column_stack([xgb_oof, lgb_oof])
meta_test = np.column_stack([
    xgb_model.predict_proba(X_test)[:, 1],
    lgb_model.predict_proba(X_test)[:, 1]
])

scaler = StandardScaler()
meta_train_s = scaler.fit_transform(meta_train)
meta_test_s = scaler.transform(meta_test)

meta_model = LogisticRegression(C=1.0, random_state=SEED)
meta_model.fit(meta_train_s, y_train)

models_preds = {
    'XGBoost':  xgb_model.predict_proba(X_test)[:, 1],
    'LightGBM': lgb_model.predict_proba(X_test)[:, 1],
    'Ensemble': meta_model.predict_proba(meta_test_s)[:, 1],
}

results = {}
for name, probs in models_preds.items():
    preds = (probs >= 0.5).astype(int)
    results[name] = {
        'AUPRC':     round(average_precision_score(y_test, probs), 4),
        'AUROC':     round(roc_auc_score(y_test, probs), 4),
        'F1':        round(f1_score(y_test, preds), 4),
        'Precision': round((preds & y_test.values).sum() / max(preds.sum(), 1), 4),
        'Recall':    round((preds & y_test.values).sum() / max(y_test.sum(), 1), 4),
    }
    print(f"\n{name}: AUPRC={results[name]['AUPRC']} | AUROC={results[name]['AUROC']} | F1={results[name]['F1']}")

palette = {'XGBoost': '#e74c3c', 'LightGBM': '#3498db', 'Ensemble': '#2ecc71'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Fraud Detection Model Evaluation', fontsize=15, fontweight='bold')

ax = axes[0]
for name, probs in models_preds.items():
    p, r, _ = precision_recall_curve(y_test, probs)
    ax.plot(r, p, label=f"{name} (AUPRC={results[name]['AUPRC']})", color=palette[name], lw=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
for name, probs in models_preds.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['AUROC']})", color=palette[name], lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
ens_preds = (models_preds['Ensemble'] >= 0.5).astype(int)
cm = confusion_matrix(y_test, ens_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
ax.set_title('Ensemble Confusion Matrix')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
feat_imp = pd.Series(xgb_model.feature_importances_, index=available)
feat_imp.nlargest(20).sort_values().plot(kind='barh', color='#3498db', ax=ax)
ax.set_title('Top 20 Feature Importances (XGBoost)', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

sample = X_test.sample(500, random_state=SEED)
explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(sample)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_vals, sample, plot_type='bar', max_display=15, show=False)
plt.title('SHAP Feature Importance (Mean |SHAP Value|)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, sample, max_display=15, show=False)
plt.title('SHAP Beeswarm — Feature Impact on Fraud Score', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

ens_scores = models_preds['Ensemble']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, color, name in [(0, '#3498db', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
    axes[0].hist(ens_scores[y_test == label], bins=50, alpha=0.7, color=color, label=name, density=True)
axes[0].set_xlabel('Fraud Probability Score')
axes[0].set_ylabel('Density')
axes[0].set_title('Score Distribution by Class')
axes[0].legend()
axes[0].grid(alpha=0.3)

thresholds = np.linspace(0.1, 0.9, 50)
prec_list, rec_list, f1_list = [], [], []
for t in thresholds:
    pp = (ens_scores >= t).astype(int)
    prec_list.append((pp & y_test.values).sum() / max(pp.sum(), 1))
    rec_list.append((pp & y_test.values).sum() / max(y_test.sum(), 1))
    f1_list.append(f1_score(y_test, pp, zero_division=0))

axes[1].plot(thresholds, prec_list, label='Precision', color='#3498db', lw=2)
axes[1].plot(thresholds, rec_list, label='Recall', color='#e74c3c', lw=2)
axes[1].plot(thresholds, f1_list, label='F1', color='#2ecc71', lw=2)
best_t = thresholds[np.argmax(f1_list)]
axes[1].axvline(best_t, color='gray', linestyle='--', label=f'Best threshold={best_t:.2f}')
axes[1].set_xlabel('Decision Threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Threshold Analysis')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/score_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

joblib.dump(xgb_model,  f'{MDIR}/xgb_model.pkl')
joblib.dump(lgb_model,  f'{MDIR}/lgb_model.pkl')
joblib.dump(meta_model, f'{MDIR}/meta_model.pkl')
joblib.dump(scaler,     f'{MDIR}/scaler.pkl')
joblib.dump(available,  f'{MDIR}/feature_cols.pkl')

pd.DataFrame(results).T.to_csv(f'{OUTDIR}/model_results.csv')

print(f"\nTraining complete. Ensemble AUPRC: {results['Ensemble']['AUPRC']} | AUROC: {results['Ensemble']['AUROC']}")
