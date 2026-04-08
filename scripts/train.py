#!/usr/bin/env python
import argparse
from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='cnn_simple')
    parser.add_argument('--data_folder', default=None)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--model_path', default='models/model.keras')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--experiment_name', default='exp')
    args = parser.parse_args()
    train(
        arch=args.arch,
        data_folder=args.data_folder,
        window_size=args.window_size,
        step=args.step,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        l2_reg=args.l2_reg,
        experiment_name=args.experiment_name
    )