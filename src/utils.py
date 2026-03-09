import numpy as np
import matplotlib.pyplot as plt

def normalize_windows(X):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i, w in enumerate(X):
        mean = np.mean(w)
        std = np.std(w)
        X_norm[i] = (w - mean) / std if std != 0 else w
    return X_norm

def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.legend()
    plt.savefig(save_path)