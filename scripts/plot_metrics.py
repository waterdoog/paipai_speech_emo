#!/usr/bin/env python3
"""Plot train/val/test loss and accuracy curves."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_history(history, title, output_path):
    epochs = [item["epoch"] for item in history]
    losses = [item.get("loss") for item in history]
    accs = [item.get("accuracy") for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, accs, marker="o")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_single(metrics, title, output_path):
    loss = metrics.get("loss")
    acc = metrics.get("accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))
    axes[0].bar([0], [loss])
    axes[0].set_title("Loss")
    axes[0].set_xticks([])

    axes[1].bar([0], [acc])
    axes[1].set_title("Accuracy")
    axes[1].set_xticks([])

    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--output_dir", default="outputs/plots")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)

    train_path = metrics_dir / "train_history.json"
    if train_path.exists():
        plot_history(load_json(train_path), "Train Metrics", output_dir / "train_metrics.png")

    val_path = metrics_dir / "val_history.json"
    if val_path.exists():
        plot_history(load_json(val_path), "Val Metrics", output_dir / "val_metrics.png")

    test_path = metrics_dir / "test_metrics.json"
    if test_path.exists():
        plot_single(load_json(test_path), "Test Metrics", output_dir / "test_metrics.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
