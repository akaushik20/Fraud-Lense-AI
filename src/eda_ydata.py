import os
import sys
from ydata_profiling import ProfileReport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helper import load_ieee_cis, reduce_mem_usage

OUTPUT_DIR = 'outputs/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    print('Loading data...')
    df = load_ieee_cis()
    
    # Sample 50k rows for profiling to reduce memory usage
    print(f'Sampling 50,000 rows from {len(df):,} total...')
    df_sample = df.sample(n=200000, random_state=42)
    df_sample = reduce_mem_usage(df_sample)

    print('Generating profiling report (minimal mode for speed)...')
    profile = ProfileReport(
        df_sample,
        title='Fraud Detection EDA Report (50k sample)',
        minimal=True,
        explorative=False
    )
    
    out_path = os.path.join(OUTPUT_DIR, 'fraud_profile_report.html')
    profile.to_file(out_path)
    print(f'Report saved to {out_path}')
    
    # Uncomment to run on full dataset (not recommended due to memory constraints)
    # print(f'\nGenerating report for full dataset ({len(df):,} rows)...')
    # profile_full = ProfileReport(df, title='Fraud Detection EDA Report (Full)', minimal=True)
    # profile_full.to_file(os.path.join(OUTPUT_DIR, 'fraud_profile_report_full.html'))
