import os
import sys
import sweetviz as sv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helper import load_ieee_cis, reduce_mem_usage

OUTPUT_DIR = 'outputs/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    print('Loading data...')
    df = load_ieee_cis()
    df = reduce_mem_usage(df)

    print('Generating Sweetviz report (this may take a few minutes)...')
    report = sv.analyze(df, target_feat='isFraud')
    # workaround for sweetviz 2.3.x bug — attribute not initialized on some builds
    if not hasattr(report, 'associations_html_source'):
        report.associations_html_source = None
    out_path = os.path.join(OUTPUT_DIR, 'sweetviz_report.html')
    report.show_html(out_path, open_browser=False)
    print(f'Report saved to {out_path}')
