import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from . import models, utils, data_loader
from .config import Config

def train(arch='cnn_simple', data_folder=None, window_size=None, step=None,
          model_path='best_model.keras', epochs=None, batch_size=None, lr=None):
    # Используем значения из Config, если не переданы явно
    data_folder = data_folder or Config.DATA_FOLDER
    window_size = window_size or Config.WINDOW_SIZE
    step = step or Config.STEP
    epochs = epochs or Config.EPOCHS
    batch_size = batch_size or Config.BATCH_SIZE
    lr = lr or Config.LEARNING_RATE

    print("Загрузка данных...")
    X, y = data_loader.load_data_from_folder(data_folder, window_size, step)
    print(f"Всего окон: {X.shape[0]}")
    X_norm = utils.normalize_windows(X)
    X_norm = X_norm[..., np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.2, stratify=y, random_state=42
    )

    build_fn = getattr(models, f'build_{arch}')
    model = build_fn(input_shape=(window_size, 1), learning_rate=lr)

    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=1)

    # Сохраняем метаданные
    meta = {'window_size': window_size, 'step': step, 'arch': arch, 'class_names': Config.CLASS_NAMES}
    with open(model_path.replace('.keras', '_meta.json'), 'w') as f:
        json.dump(meta, f)

    utils.plot_history(history, model_path.replace('.keras', '_history.png'))
    print(f"Модель сохранена в {model_path}")