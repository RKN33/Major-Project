import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = '/home/claude/fraud_detection/outputs'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv('/home/claude/fraud_detection/data/transactions.csv')
df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24

print(f"Dataset: {len(df):,} rows | Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Exploratory Data Analysis — Fraud Detection Dataset', fontsize=14, fontweight='bold')

ax = axes[0, 0]
counts = df['isFraud'].value_counts()
ax.bar(['Legitimate', 'Fraud'], counts.values, color=['#3498db', '#e74c3c'], edgecolor='white')
ax.set_title('Class Distribution')
ax.set_ylabel('Count')
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)

ax = axes[0, 1]
for label, color, name in [(0, '#3498db', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
    vals = np.log1p(df[df['isFraud'] == label]['TransactionAmt'])
    ax.hist(vals, bins=50, alpha=0.65, color=color, label=name, density=True)
ax.set_title('Log(Transaction Amount) Distribution')
ax.set_xlabel('log(1 + Amount)')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 2]
fraud_by_hour = df.groupby('hour_of_day')['isFraud'].mean() * 100
ax.bar(fraud_by_hour.index, fraud_by_hour.values, color='#e67e22', edgecolor='white')
ax.set_title('Fraud Rate by Hour of Day')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Fraud Rate (%)')
ax.grid(alpha=0.3)

ax = axes[1, 0]
dom_fraud = df.groupby('P_emaildomain')['isFraud'].mean().sort_values(ascending=False)
colors_ = ['#e74c3c' if r > 0.1 else '#3498db' for r in dom_fraud.values]
ax.barh(dom_fraud.index, dom_fraud.values * 100, color=colors_)
ax.set_title('Fraud Rate by Email Domain')
ax.set_xlabel('Fraud Rate (%)')
ax.grid(alpha=0.3)

ax = axes[1, 1]
prod_fraud = df.groupby('ProductCD')['isFraud'].agg(['mean', 'count'])
prod_fraud['mean'] *= 100
ax.bar(prod_fraud.index, prod_fraud['mean'], color='#9b59b6', edgecolor='white')
ax.set_title('Fraud Rate by Product Category')
ax.set_xlabel('Product Code')
ax.set_ylabel('Fraud Rate (%)')
ax.grid(alpha=0.3)

ax = axes[1, 2]
addr_fraud = df.groupby('addr_match')['isFraud'].mean() * 100
ax.bar(['Mismatch', 'Match'], addr_fraud.values, color=['#e74c3c', '#2ecc71'], edgecolor='white')
ax.set_title('Fraud Rate: Address Match vs Mismatch')
ax.set_ylabel('Fraud Rate (%)')
ax.grid(alpha=0.3)
for i, v in enumerate(addr_fraud.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
num_cols = ['TransactionAmt', 'hour_of_day', 'isFraud']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("EDA complete. Plots saved to outputs/")
