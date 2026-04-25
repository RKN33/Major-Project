import numpy as np
import pandas as pd

np.random.seed(42)

N = 50000
FRAUD_RATE = 0.035


def generate_dataset(n=N, fraud_rate=FRAUD_RATE):
    n_fraud = int(n * fraud_rate)

    timestamps = np.sort(np.random.uniform(86400, 86400 + 180 * 86400, n))

    n_cards = int(n * 0.15)
    n_devices = int(n * 0.25)
    card_ids = np.random.randint(1000, 1000 + n_cards, n)
    device_ids = np.random.randint(5000, 5000 + n_devices, n)

    categories = np.random.choice(['W', 'H', 'C', 'S'], size=n, p=[0.45, 0.20, 0.25, 0.10])

    amounts = np.where(
        categories == 'W', np.random.lognormal(4.5, 1.2, n),
        np.where(
            categories == 'H', np.random.lognormal(3.8, 0.9, n),
            np.where(categories == 'C', np.random.lognormal(3.2, 0.8, n),
                     np.random.lognormal(2.5, 0.7, n))
        )
    ).round(2)

    email_domains = np.random.choice(
        ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
         'protonmail.com', 'anonymous.com', 'temp-mail.org'],
        size=n, p=[0.35, 0.25, 0.15, 0.10, 0.05, 0.06, 0.04]
    )

    card_types = np.random.choice(['credit', 'debit'], n, p=[0.55, 0.45])
    card_banks = np.random.choice(
        ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'Other'],
        n, p=[0.20, 0.20, 0.18, 0.15, 0.12, 0.15]
    )

    addr_match = np.random.choice([0, 1], n, p=[0.12, 0.88])

    device_types = np.random.choice(
        ['desktop', 'mobile', 'tablet', 'unknown'],
        n, p=[0.38, 0.45, 0.10, 0.07]
    )

    df = pd.DataFrame({
        'TransactionID': np.arange(100000, 100000 + n),
        'TransactionDT': timestamps,
        'TransactionAmt': amounts,
        'ProductCD': categories,
        'card_id': card_ids,
        'device_id': device_ids,
        'card_type': card_types,
        'card_bank': card_banks,
        'P_emaildomain': email_domains,
        'addr_match': addr_match,
        'DeviceType': device_types,
        'isFraud': 0
    })

    fraud_idx = np.random.choice(df.index, size=n_fraud, replace=False)
    df.loc[fraud_idx, 'isFraud'] = 1

    susp_mask = df.index.isin(fraud_idx)

    change_mask = susp_mask & (np.random.rand(n) < 0.45)
    df.loc[change_mask, 'P_emaildomain'] = np.random.choice(
        ['anonymous.com', 'temp-mail.org'], size=change_mask.sum()
    )

    small_fraud = fraud_idx[:n_fraud // 3]
    large_fraud = fraud_idx[n_fraud // 3: 2 * n_fraud // 3]
    df.loc[small_fraud, 'TransactionAmt'] = np.random.uniform(0.5, 2.5, len(small_fraud)).round(2)
    df.loc[large_fraud, 'TransactionAmt'] = np.random.lognormal(6.5, 0.8, len(large_fraud)).round(2)

    night_fraud = fraud_idx[2 * n_fraud // 3:]
    df.loc[night_fraud, 'TransactionDT'] = (
        df.loc[night_fraud, 'TransactionDT']
        .apply(lambda x: (x % 86400) + np.random.uniform(7200, 18000))
    )

    addr_change = susp_mask & (np.random.rand(n) < 0.60)
    df.loc[addr_change, 'addr_match'] = 0

    df = df.sort_values('TransactionDT').reset_index(drop=True)
    return df


if __name__ == '__main__':
    import os
    os.makedirs('/home/claude/fraud_detection/data', exist_ok=True)
    df = generate_dataset()
    df.to_csv('/home/claude/fraud_detection/data/transactions.csv', index=False)
    print(f"Dataset: {len(df):,} rows | Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.1f}%)")
