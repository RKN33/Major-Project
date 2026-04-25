import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap

from feature_engineering import engineer_features

MDIR = '/home/claude/fraud_detection/models'


def load_models():
    return {
        'xgb':      joblib.load(f'{MDIR}/xgb_model.pkl'),
        'lgb':      joblib.load(f'{MDIR}/lgb_model.pkl'),
        'meta':     joblib.load(f'{MDIR}/meta_model.pkl'),
        'scaler':   joblib.load(f'{MDIR}/scaler.pkl'),
        'features': joblib.load(f'{MDIR}/feature_cols.pkl'),
    }


def score_transactions(df_raw: pd.DataFrame, models: dict) -> pd.DataFrame:
    t0 = time.perf_counter()

    df_feat = engineer_features(df_raw)
    avail = models['features']
    X = df_feat[[c for c in avail if c in df_feat.columns]].fillna(0)

    xgb_p = models['xgb'].predict_proba(X)[:, 1]
    lgb_p = models['lgb'].predict_proba(X)[:, 1]

    meta_in = models['scaler'].transform(np.column_stack([xgb_p, lgb_p]))
    fraud_score = models['meta'].predict_proba(meta_in)[:, 1]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    df_raw = df_raw.copy()
    df_raw['fraud_score'] = fraud_score.round(4)
    df_raw['xgb_score'] = xgb_p.round(4)
    df_raw['lgb_score'] = lgb_p.round(4)
    df_raw['risk_level'] = pd.cut(
        fraud_score,
        bins=[0, 0.3, 0.6, 0.85, 1.001],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )
    df_raw['flag'] = (fraud_score >= 0.5).astype(int)
    df_raw['inference_ms'] = round(elapsed_ms / len(df_raw), 2)
    return df_raw


def explain_transaction(df_raw: pd.DataFrame, idx: int, models: dict):
    df_feat = engineer_features(df_raw)
    avail = models['features']
    X = df_feat[[c for c in avail if c in df_feat.columns]].fillna(0)

    explainer = shap.TreeExplainer(models['xgb'])
    shap_vals = explainer.shap_values(X.iloc[[idx]])
    row = X.iloc[idx]

    contrib = pd.Series(shap_vals[0], index=X.columns).sort_values(key=abs, ascending=False)
    print(f"\n{'='*55}")
    print(f"SHAP Explanation — Transaction #{df_raw.iloc[idx].get('TransactionID', idx)}")
    print(f"{'='*55}")
    print(f"{'Feature':<35} {'SHAP Value':>10}  Direction")
    print("-" * 60)
    for feat, val in contrib.head(10).items():
        direction = 'FRAUD' if val > 0 else 'LEGIT'
        print(f"  {feat:<33} {val:>+10.4f}  {direction}  (raw={row[feat]:.3f})")
    print(f"{'='*55}\n")


def demo(models):
    examples = pd.DataFrame([
        dict(TransactionID=1, TransactionDT=3600*14,   TransactionAmt=45.99,
             ProductCD='C', card_id=1001, device_id=5001,
             card_type='debit',  card_bank='HDFC',
             P_emaildomain='gmail.com', addr_match=1, DeviceType='mobile'),
        dict(TransactionID=2, TransactionDT=3600*3,    TransactionAmt=4899.00,
             ProductCD='W', card_id=1002, device_id=5002,
             card_type='credit', card_bank='Other',
             P_emaildomain='temp-mail.org', addr_match=0, DeviceType='unknown'),
        dict(TransactionID=3, TransactionDT=3600*3+60, TransactionAmt=1.00,
             ProductCD='W', card_id=1002, device_id=5002,
             card_type='credit', card_bank='Other',
             P_emaildomain='temp-mail.org', addr_match=0, DeviceType='unknown'),
        dict(TransactionID=4, TransactionDT=86400*6+3600*19, TransactionAmt=120.00,
             ProductCD='H', card_id=1003, device_id=5003,
             card_type='credit', card_bank='ICICI',
             P_emaildomain='yahoo.com', addr_match=1, DeviceType='desktop'),
    ])

    scored = score_transactions(examples, models)

    print("\n" + "=" * 65)
    print("REAL-TIME FRAUD SCORING DEMO")
    print("=" * 65)
    for _, row in scored.iterrows():
        status = "FLAGGED" if row['flag'] == 1 else "CLEARED"
        print(f"\n  TxnID={row['TransactionID']} | Amt={row['TransactionAmt']:>8.2f} "
              f"| Score={row['fraud_score']:.4f} | Risk={row['risk_level']} | {status}")
        print(f"    XGB={row['xgb_score']:.4f}  LGB={row['lgb_score']:.4f}  "
              f"Latency={row['inference_ms']} ms/tx")
    print("\n" + "=" * 65)

    explain_transaction(scored.iloc[[1]].reset_index(drop=True), 0, models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=str, help='CSV file for batch scoring')
    args = parser.parse_args()

    models = load_models()

    if args.batch:
        df = pd.read_csv(args.batch)
        scored = score_transactions(df, models)
        out = args.batch.replace('.csv', '_scored.csv')
        scored.to_csv(out, index=False)
        n_flagged = scored['flag'].sum()
        print(f"Scored {len(df):,} transactions | Flagged: {n_flagged:,} ({n_flagged/len(df)*100:.2f}%)")
        print(f"Saved to {out}")
    else:
        demo(models)
