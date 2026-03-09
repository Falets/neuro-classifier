import json
import numpy as np
import tensorflow as tf
from . import data_loader, utils
from .config import Config

def predict_one_file(file_path, model_path):
    model = tf.keras.models.load_model(model_path)
    meta_path = model_path.replace('.keras', '_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    window_size = meta['window_size']
    step = meta.get('step', window_size // 5)
    class_names = meta.get('class_names', Config.CLASS_NAMES)

    values = data_loader.read_csv_signal(file_path)
    if len(values) < window_size:
        values = np.pad(values, (0, window_size - len(values)), 'constant')
        windows = values.reshape(1, -1)
    else:
        windows = data_loader.create_windows(values, window_size, step)

    windows_norm = utils.normalize_windows(windows)
    windows_norm = windows_norm[..., np.newaxis]

    probs = model.predict(windows_norm, verbose=0)
    avg_prob = np.mean(probs, axis=0)
    pred = np.argmax(avg_prob)
    conf = avg_prob[pred]
    return class_names[pred], conf