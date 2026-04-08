import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import callbacks
from . import models, utils, data_loader
from .config import Config
from tensorflow import keras
from pathlib import Path
import datetime



def train(arch='cnn_deeper', data_folder=None, window_size=None, step=None,
          model_path='best_model.keras', epochs=None, batch_size=None, lr=None,
          dropout=0.3, l2_reg=1e-4, experiment_name=None):

    data_folder = data_folder or Config.DATA_FOLDER
    window_size = window_size or Config.WINDOW_SIZE
    step = step or Config.STEP
    epochs = epochs or Config.EPOCHS
    batch_size = batch_size or Config.BATCH_SIZE
    lr = lr or Config.LEARNING_RATE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_name or f"{arch}_{timestamp}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(experiment_dir / Path(model_path).name)

    print("Загрузка данных...")
    X, y = data_loader.load_data_from_folder(data_folder, window_size, step)
    print(f"Всего окон: {X.shape[0]}")

    X_norm = utils.normalize_windows(X)
    X_norm = X_norm[..., np.newaxis]

    # 1) train + temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y, test_size=0.3, stratify=y, random_state=42
    )

    # 2) val + test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    build_fn = getattr(models, f'build_{arch}')
    model = build_fn(
        input_shape=(window_size, 1),
        learning_rate=lr,
        l2_reg=l2_reg,
        dropout_rate=dropout
    )

    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    model = keras.models.load_model(model_path)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    test_acc = np.mean(y_pred == y_test)
    results = {
        "arch": arch,
        "window_size": window_size,
        "step": step,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
    }

    with open(experiment_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    meta = {
        'window_size': window_size,
        'step': step,
        'arch': arch,
        'class_names': Config.CLASS_NAMES
    }
    with open(model_path.replace('.keras', '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    utils.plot_history(history, str(experiment_dir / "history.png"))
    utils.plot_confusion(y_test, y_pred, Config.CLASS_NAMES, str(experiment_dir / "cm.png"))
    utils.save_classification_report(y_test, y_pred, Config.CLASS_NAMES, str(experiment_dir / "report.txt"))

    print(f"Модель сохранена в {model_path}")
    return model, history, (X_test, y_test, y_pred)