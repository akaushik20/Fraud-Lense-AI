import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helper import load_ieee_cis, reduce_mem_usage

OUTPUT_DIR = 'outputs/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def write_summary(lines, mode='a'):
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), mode) as f:
        f.write('\n'.join(lines) + '\n')


def dataset_overview(df):
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    fraud_rate = df.isFraud.mean()
    lines = [
        '=== Dataset Overview ===',
        f'Rows:          {len(df):,}',
        f'Columns:       {df.shape[1]}',
        f'Memory usage:  {mem_mb:.1f} MB',
        f'Fraud rate:    {fraud_rate:.2%}  ({df.isFraud.sum():,} fraudulent)',
        f'Dtype counts:  {dict(df.dtypes.value_counts())}',
    ]
    write_summary(lines, mode='w')
    print('\n'.join(lines))


def class_distribution(df):
    counts = df.isFraud.value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Legitimate', 'Fraud'], counts.values, color=['steelblue', 'tomato'])
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.01, f'{v:,}', ha='center', fontsize=10)
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    save(fig, 'class_distribution.png')


def missing_values(df):
    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss_df = miss.reset_index()
    miss_df.columns = ['feature', 'pct_missing']
    miss_df.to_csv(os.path.join(OUTPUT_DIR, 'missing_values.csv'), index=False)

    top30 = miss_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top30, x='pct_missing', y='feature', ax=ax, color='steelblue')
    ax.set_title('Top 30 Features by % Missing')
    ax.set_xlabel('% Missing')
    save(fig, 'missing_values.png')
    print(f'Saved outputs/eda/missing_values.csv')


def transaction_amount(df):
    fraud = df[df.isFraud == 1]['TransactionAmt']
    legit = df[df.isFraud == 0]['TransactionAmt']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(legit), bins=80, alpha=0.6, label='Legitimate', color='steelblue')
    ax.hist(np.log1p(fraud), bins=80, alpha=0.6, label='Fraud', color='tomato')
    ax.set_xlabel('log(1 + TransactionAmt)')
    ax.set_ylabel('Count')
    ax.set_title('Transaction Amount Distribution (log scale)')
    ax.legend()
    save(fig, 'transaction_amt_dist.png')

    lines = [
        '',
        '=== TransactionAmt by Class ===',
        f'Legitimate — mean: ${legit.mean():.2f}  median: ${legit.median():.2f}',
        f'Fraud      — mean: ${fraud.mean():.2f}  median: ${fraud.median():.2f}',
    ]
    write_summary(lines)
    print('\n'.join(lines))


def categorical_fraud_rates(df):
    cats = ['ProductCD', 'card4', 'card6', 'DeviceType']
    cats = [c for c in cats if c in df.columns]

    fig, axes = plt.subplots(1, len(cats), figsize=(4 * len(cats), 5))
    if len(cats) == 1:
        axes = [axes]

    for ax, col in zip(axes, cats):
        rates = df.groupby(col)['isFraud'].mean().sort_values(ascending=False)
        sns.barplot(x=rates.index, y=rates.values, ax=ax, palette='coolwarm')
        ax.set_title(f'Fraud Rate by {col}')
        ax.set_ylabel('Fraud Rate')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Fraud Rate by Categorical Features', y=1.02)
    save(fig, 'categorical_fraud_rates.png')


def email_domain_fraud_rates(df):
    if 'P_emaildomain' not in df.columns:
        return
    top_domains = df['P_emaildomain'].value_counts().head(15).index
    subset = df[df['P_emaildomain'].isin(top_domains)]
    rates = subset.groupby('P_emaildomain')['isFraud'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=rates.index, y=rates.values, ax=ax, palette='coolwarm')
    ax.set_title('Fraud Rate by Purchaser Email Domain (Top 15)')
    ax.set_ylabel('Fraud Rate')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    save(fig, 'email_domain_fraud_rates.png')


def fraud_over_time(df):
    if 'TransactionDT' not in df.columns:
        return
    df = df.copy()
    df['time_bin'] = pd.cut(df['TransactionDT'], bins=50)
    rates = df.groupby('time_bin', observed=True)['isFraud'].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(rates)), rates.values, color='tomato', linewidth=1.5)
    ax.set_title('Fraud Rate Over Time (TransactionDT bins)')
    ax.set_ylabel('Fraud Rate')
    ax.set_xlabel('Time Bin (earliest → latest)')
    save(fig, 'fraud_over_time.png')

    lines = ['', '=== Time Patterns ===',
             'Fraud rate over time saved — use TimeSeriesSplit to prevent future leakage in CV.']
    write_summary(lines)


def cd_feature_distributions(df):
    features = [f for f in ['C1', 'C2', 'D1', 'D4'] if f in df.columns]
    if not features:
        return

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
    if len(features) == 1:
        axes = [axes]

    for ax, col in zip(axes, features):
        fraud_vals = df[df.isFraud == 1][col].dropna()
        legit_vals = df[df.isFraud == 0][col].dropna()
        # clip to 99th percentile to avoid extreme outlier distortion
        upper = df[col].quantile(0.99)
        ax.boxplot(
            [legit_vals.clip(upper=upper), fraud_vals.clip(upper=upper)],
            patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.6),
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Legit', 'Fraud'])
        ax.set_title(col)
        ax.set_ylabel('Value (clipped at 99th pct)')

    fig.suptitle('C/D Feature Distributions by Fraud Label')
    save(fig, 'cd_features_boxplot.png')


def top_correlations(df):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()['isFraud'].drop('isFraud').abs().sort_values(ascending=False).head(20)
    corr_df = corr.reset_index()
    corr_df.columns = ['feature', 'abs_correlation']
    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'top_correlations.csv'), index=False)
    print('Saved outputs/eda/top_correlations.csv')

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(data=corr_df, x='abs_correlation', y='feature', ax=ax, color='steelblue')
    ax.set_title('Top 20 Features Correlated with isFraud')
    ax.set_xlabel('|Correlation|')
    save(fig, 'top_correlations.png')


if __name__ == '__main__':
    print('Loading data...')
    df = load_ieee_cis()
    df = reduce_mem_usage(df)

    print('\nRunning EDA...')
    dataset_overview(df)
    class_distribution(df)
    missing_values(df)
    transaction_amount(df)
    categorical_fraud_rates(df)
    email_domain_fraud_rates(df)
    fraud_over_time(df)
    cd_feature_distributions(df)
    top_correlations(df)

    print(f'\nAll outputs saved to {OUTPUT_DIR}/')
