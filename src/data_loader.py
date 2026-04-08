import os
import numpy as np
import pandas as pd


def read_signal_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, header=None)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    if 'value' in df.columns:
        return df['value'].values.astype(np.float32)

    if len(df.columns) >= 2:
        return df.iloc[:, 1].values.astype(np.float32)

    col = df.iloc[:, 0].astype(str)

    if "," in col.iloc[1]:
        values = []
        for row in col.iloc[1:]:
            try:
                parts = row.split(",")
                values.append(float(parts[1]))
            except Exception:
                continue
        return np.array(values, dtype=np.float32)

    raise ValueError(f"Файл {file_path} не содержит корректных данных сигнала")


def create_windows(values, window_size, step):
    windows = []
    for start in range(0, len(values) - window_size + 1, step):
        windows.append(values[start:start + window_size])
    return np.array(windows, dtype=np.float32)


def load_data_from_folder(folder_path, window_size, step):
    X, y = [], []

    for file in os.listdir(folder_path):
        if not (file.endswith('.csv') or file.endswith('.xlsx') or file.endswith('.xls')):
            continue

        path = os.path.join(folder_path, file)
        values = read_signal_file(path)
        windows = create_windows(values, window_size, step)

        name = file.lower()
        if 'epsp' in name:
            label = 0
        elif 'ps' in name:
            label = 1
        else:
            raise ValueError(f"Класс не определён в имени файла {file}")

        X.append(windows)
        y.append(np.full(len(windows), label))

    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y