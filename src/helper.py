import pandas as pd
from sklearn.preprocessing import LabelEncoder

INTERPRETABLE_FEATURES = [
    'TransactionAmt', 'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
    'DeviceType', 'DeviceInfo',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
]


def load_ieee_cis(data_dir='data'):
    txn = pd.read_csv(f'{data_dir}/train_transaction.csv')
    txn = reduce_mem_usage(txn)
    idn = pd.read_csv(f'{data_dir}/train_identity.csv')
    idn = reduce_mem_usage(idn)
    df = txn.merge(idn, on='TransactionID', how='left')
    print(f'Loaded {len(df):,} transactions | Fraud rate: {df.isFraud.mean():.1%}')
    return df


def reduce_mem_usage(df):
    for col in df.select_dtypes('int').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes('float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def encode_categoricals(df, encoders=None):
    df = df.copy()
    obj_cols = df.select_dtypes('object').columns.tolist()
    fitted = {}

    if encoders is None:
        for col in obj_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            fitted[col] = le
        return df, fitted
    else:
        for col in obj_cols:
            if col in encoders:
                le = encoders[col]
                # unseen labels at inference time get -1
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df[col] = -1
        return df, encoders
