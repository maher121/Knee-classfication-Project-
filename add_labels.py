import os
import pandas as pd

FEATURES_DIR = 'final_Features'
LABEL_COL = 'label'


def get_label_source(files):
    sources = []
    for name in files:
        path = os.path.join(FEATURES_DIR, name)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if LABEL_COL in header:
            sources.append(name)
    return sources


def load_labels(source_name, expected_len=None):
    path = os.path.join(FEATURES_DIR, source_name)
    df = pd.read_csv(path, usecols=[LABEL_COL])
    labels = df[LABEL_COL]
    if expected_len is None:
        expected_len = len(labels)
    elif len(labels) != expected_len:
        raise ValueError(
            f'{source_name}: {len(labels)} labels, expected {expected_len}'
        )
    return labels


def main():
    csv_files = sorted(
        f for f in os.listdir(FEATURES_DIR) if f.lower().endswith('.csv')
    )
    if not csv_files:
        print(f'No CSV files found in {FEATURES_DIR}')
        return

    sources = get_label_source(csv_files)
    if not sources:
        print(f'No CSV containing a "{LABEL_COL}" column found.')
        return

    print(f'Label sources found: {sources}')
    labels = load_labels(sources[0])
    n = len(labels)

    for src in sources[1:]:
        other = load_labels(src, expected_len=n)
        if not other.equals(labels):
            print(f'WARNING: labels differ between {sources[0]} and {src}!')

    targets = [
        f for f in csv_files
        if LABEL_COL not in pd.read_csv(
            os.path.join(FEATURES_DIR, f), nrows=0
        ).columns
    ]

    if not targets:
        print('All CSV files already contain a label column. Nothing to do.')
        return

    for name in targets:
        path = os.path.join(FEATURES_DIR, name)
        df = pd.read_csv(path)
        if len(df) != n:
            print(
                f'SKIPPED {name}: {len(df)} rows does not match '
                f'{n} labels.'
            )
            continue
        if LABEL_COL in df.columns:
            print(f'SKIPPED {name}: already has "{LABEL_COL}".')
            continue
        df[LABEL_COL] = labels.values
        backup = path + '.bak'
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(path, backup)
        try:
            df.to_csv(path, index=False)
            os.remove(backup)
            print(f'ADDED label to {name} ({df.shape[1]} columns now)')
        except Exception as e:
            os.rename(backup, path)
            print(f'FAILED {name}: {e} (original restored)')


if __name__ == '__main__':
    main()
