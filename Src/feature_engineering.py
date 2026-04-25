import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('TransactionDT').reset_index(drop=True)

    df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
    df['is_night'] = df['hour_of_day'].between(2, 5).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['log_amount'] = np.log1p(df['TransactionAmt'])
    df['amt_cents'] = (df['TransactionAmt'] % 1).round(2)
    df['is_round_amt'] = (df['amt_cents'] == 0).astype(int)

    for window, label in [(10, '10'), (50, '50'), (200, '200')]:
        grp = df.groupby('card_id')['TransactionAmt']
        df[f'card_amt_mean_{label}'] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'card_amt_std_{label}'] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )
        df[f'card_tx_count_{label}'] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).count()
        )

    df['amt_dev_mean_50'] = (df['TransactionAmt'] - df['card_amt_mean_50']).abs().fillna(0)
    df['amt_zscore_50'] = np.where(
        df['card_amt_std_50'] > 0,
        (df['TransactionAmt'] - df['card_amt_mean_50']) / df['card_amt_std_50'],
        0
    )

    df['card_velocity'] = df.groupby('card_id').cumcount()
    df['device_velocity'] = df.groupby('device_id').cumcount()

    df['prev_ts'] = df.groupby('card_id')['TransactionDT'].shift(1)
    df['time_since_last_card_tx'] = (df['TransactionDT'] - df['prev_ts']).fillna(999999)
    df.drop(columns=['prev_ts'], inplace=True)

    df['is_suspicious_email'] = df['P_emaildomain'].isin(
        ['anonymous.com', 'temp-mail.org', 'protonmail.com']
    ).astype(int)

    df['ProductCD_enc'] = df['ProductCD'].map({'W': 0, 'H': 1, 'C': 2, 'S': 3})
    df['DeviceType_enc'] = df['DeviceType'].map({'desktop': 0, 'mobile': 1, 'tablet': 2, 'unknown': 3})
    df['card_type_enc'] = df['card_type'].map({'credit': 0, 'debit': 1})
    df['card_bank_enc'] = df['card_bank'].map({'SBI': 0, 'HDFC': 1, 'ICICI': 2, 'Axis': 3, 'Kotak': 4, 'Other': 5})

    drop_cols = ['TransactionID', 'TransactionDT', 'ProductCD', 'DeviceType',
                 'card_type', 'card_bank', 'P_emaildomain', 'card_id', 'device_id']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


FEATURE_COLS = [
    'TransactionAmt', 'log_amount', 'amt_cents', 'is_round_amt',
    'hour_of_day', 'day_of_week', 'is_night', 'is_weekend',
    'card_amt_mean_10', 'card_amt_mean_50', 'card_amt_mean_200',
    'card_amt_std_10', 'card_amt_std_50', 'card_amt_std_200',
    'card_tx_count_10', 'card_tx_count_50', 'card_tx_count_200',
    'amt_dev_mean_50', 'amt_zscore_50',
    'card_velocity', 'device_velocity', 'time_since_last_card_tx',
    'is_suspicious_email', 'addr_match',
    'ProductCD_enc', 'DeviceType_enc', 'card_type_enc', 'card_bank_enc',
]
