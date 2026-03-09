import os
import numpy as np
import pandas as pd

def read_csv_signal(file_path):
    df = pd.read_csv(file_path)
    if 'value' in df.columns:
        return df['value'].values.astype(np.float32)
    elif len(df.columns) >= 2:
        return df.iloc[:, 1].values.astype(np.float32)
    else:
        raise ValueError(f"Файл {file_path} не содержит двух колонок")

def create_windows(values, window_size, step):
    windows = []
    for start in range(0, len(values) - window_size + 1, step):
        windows.append(values[start:start+window_size])
    return np.array(windows)

def load_data_from_folder(folder_path, window_size, step):
    X, y = [], []
    for file in os.listdir(folder_path):
        if not file.endswith('.csv'):
            continue
        path = os.path.join(folder_path, file)
        values = read_csv_signal(path)
        windows = create_windows(values, window_size, step)
        label = 0 if 'epsp' in file.lower() else 1 if 'ps' in file.lower() else None
        if label is None:
            raise ValueError(f"Класс не определён в имени {file}")
        X.append(windows)
        y.append(np.full(len(windows), label))
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y