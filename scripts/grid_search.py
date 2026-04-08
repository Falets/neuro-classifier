import itertools
import json
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train


def main():
    grid = {
        "arch": ["cnn_deeper", "lstm_deeper", "cnn_bilstm"],
        "lr": [1e-3, 3e-4],
        "batch_size": [16, 32],
        "dropout": [0.2, 0.4],
        "l2_reg": [1e-4],
        "window_size": [500],
        "step": [100],
    }

    keys = list(grid.keys())
    values = list(grid.values())

    results = []
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        exp_name = (
            f"{params['arch']}_lr{params['lr']}"
            f"_bs{params['batch_size']}_do{params['dropout']}"
        )

        print(f"\n=== RUN: {exp_name} ===")
        res = train(
            arch=params["arch"],
            data_folder="./data",
            window_size=params["window_size"],
            step=params["step"],
            epochs=30,
            batch_size=params["batch_size"],
            lr=params["lr"],
            dropout=params["dropout"],
            l2_reg=params["l2_reg"],
            experiment_name=exp_name,
            model_path="model.keras",
        )

        row = {**params, **res}
        results.append(row)

        df = pd.DataFrame(results)
        df.to_csv(out_dir / "grid_search_results.csv", index=False, encoding="utf-8")

    df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    df.to_csv(out_dir / "grid_search_results_sorted.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(12, 5))
    labels = [
        f"{r['arch']}\nlr={r['lr']}\nbs={r['batch_size']}\ndo={r['dropout']}"
        for _, r in df.iterrows()
    ]
    plt.bar(range(len(df)), df["test_accuracy"])
    plt.xticks(range(len(df)), labels, rotation=45, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Comparison of experiments")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=300)
    plt.close()

    print("\nTop 5 experiments:")
    print(df.head(5)[["arch", "lr", "batch_size", "dropout", "test_accuracy", "best_val_accuracy"]])


if __name__ == "__main__":
    main()