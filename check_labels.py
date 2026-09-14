import os
import pandas as pd

FEATURES_DIR = 'final_Features'
LABEL_COL = 'label'


def check_csv_for_label(csv_path):
    df = pd.read_csv(csv_path, nrows=0)
    columns = df.columns.tolist()
    has_label = LABEL_COL in columns
    return {
        'file': csv_path,
        'n_columns': len(columns),
        'has_label': has_label,
    }


def main():
    csv_files = sorted(
        f for f in os.listdir(FEATURES_DIR) if f.lower().endswith('.csv')
    )
    if not csv_files:
        print(f'No CSV files found in {FEATURES_DIR}')
        return

    for name in csv_files:
        path = os.path.join(FEATURES_DIR, name)
        try:
            info = check_csv_for_label(path)
            status = 'HAS label' if info['has_label'] else 'NO label'
            print(f"{info['file']}: {status} ({info['n_columns']} columns)")
        except Exception as e:
            print(f'{path}: ERROR reading file ({e})')


if __name__ == '__main__':
    main()
