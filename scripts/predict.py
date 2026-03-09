#!/usr/bin/env python
import argparse
from src.predict import predict_one_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Путь к CSV-файлу с сигналом')
    parser.add_argument('--model', default='models/model.keras', help='Путь к модели')
    args = parser.parse_args()
    cls, conf = predict_one_file(args.file, args.model)
    print(f"Предсказанный класс: {cls}")
    print(f"Уверенность: {conf:.4f}")