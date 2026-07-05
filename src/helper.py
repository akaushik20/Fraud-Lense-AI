import pandas as pd
from sklearn.preprocessing import LabelEncoder

INTERPRETABLE_FEATURES = [
    'TransactionAmt', 'ProductCD',
    'P_emaildomain', 'R_emaildomain',
    'DeviceType', 'DeviceInfo',
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


def encode_categoricals(df, encoders=None, columns=None):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame to encode
        encoders: Pre-fitted encoders dict (for inference/test sets)
        columns: Explicit list of columns to encode. If None, auto-detects object dtype columns.
    
    Returns:
        df: Encoded DataFrame
        encoders: Dict of fitted LabelEncoders
    """
    df = df.copy()
    
    # Use explicit list or auto-detect
    if columns is not None:
        cols_to_encode = [c for c in columns if c in df.columns]
    else:
        cols_to_encode = df.select_dtypes('object').columns.tolist()
    
    fitted = {}

    if encoders is None:
        # Fit mode: create new encoders
        for col in cols_to_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            fitted[col] = le
        return df, fitted
    else:
        # Transform mode: use existing encoders
        for col in cols_to_encode:
            if col in encoders:
                le = encoders[col]
                # unseen labels at inference time get -1
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df[col] = -1
        return df, encoders
