#!/usr/bin/env python3
"""情感分類模型主訓練腳本

該腳本負責加載配置、構建數據集和數據加載器、初始化模型、優化器和調度器，
並執行完整的訓練流程，包括訓練、驗證和測試。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.datasets.audio_dataset import EmotionDataset, collate_batch
from src.training.loss import build_criterion, compute_class_weights
from src.training.trainer import evaluate, save_checkpoint, save_metrics, train_one_epoch
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.models.model import EmotionModel
from src.models.domain import DomainClassifier


def as_list(value):
    """將值轉換為列表形式
    
    如果值為None，返回空列表
    如果值已經是列表，直接返回
    否則將值封裝在列表中返回
    
    Args:
        value: 要轉換的值
    
    Returns:
        轉換後的列表
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_dataset(cfg, csv_path, label_map, domain_id, mode):
    """構建情感分類數據集
    
    Args:
        cfg: 配置字典
        csv_path: CSV文件路徑
        label_map: 標籤映射文件路徑
        domain_id: 領域ID
        mode: 數據集模式 (train/val/test)
    
    Returns:
        構建完成的EmotionDataset對象
    """
    data_cfg = cfg["data"]
    return EmotionDataset(
        csv_path=csv_path,
        label_list=cfg["labels"],
        label_map_path=label_map,
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        n_fft=data_cfg.get("n_fft"),
        hop_length=data_cfg.get("hop_length"),
        max_duration=data_cfg.get("max_duration", 6.0),
        min_duration=data_cfg.get("min_duration", 0.5),
        mode=mode,
        augment=cfg.get("augmentation"),
        root_dir=data_cfg.get("root_dir"),
        domain_id=domain_id,
    )


def build_train_loader(cfg):
    """構建訓練數據加載器
    
    支持多個訓練CSV文件、標籤映射和領域ID的配置
    如果提供了多個數據集，會自動進行合併
    支持設置數據集權重，用於處理不平衡數據
    
    Args:
        cfg: 配置字典
    
    Returns:
        訓練數據加載器和訓練數據集列表
    
    Raises:
        ValueError: 當配置不合法時拋出
    """
    data_cfg = cfg["data"]
    train_csvs = as_list(data_cfg.get("train_csv"))
    if not train_csvs:
        raise ValueError("train_csv is required in config")

    label_maps = as_list(data_cfg.get("train_label_maps") or data_cfg.get("label_map"))
    if len(label_maps) == 1 and len(train_csvs) > 1:
        label_maps = label_maps * len(train_csvs)
    if len(label_maps) != len(train_csvs):
        raise ValueError("label_map count must match train_csv count")

    domain_ids = as_list(data_cfg.get("train_domains") or data_cfg.get("domain_id") or 0)
    if len(domain_ids) == 1 and len(train_csvs) > 1:
        domain_ids = domain_ids * len(train_csvs)
    if len(domain_ids) != len(train_csvs):
        raise ValueError("train_domains count must match train_csv count")

    datasets = [
        build_dataset(cfg, csv_path, label_map, domain_id, "train")
        for csv_path, label_map, domain_id in zip(train_csvs, label_maps, domain_ids)
    ]

    if len(datasets) == 1:
        dataset = datasets[0]
        sampler = None
    else:
        dataset = ConcatDataset(datasets)
        weights = as_list(data_cfg.get("train_weights") or [1.0] * len(datasets))
        if len(weights) == 1 and len(datasets) > 1:
            weights = weights * len(datasets)
        if len(weights) != len(datasets):
            raise ValueError("train_weights count must match train_csv count")

        sample_weights = []
        for ds, weight in zip(datasets, weights):
            sample_weights.extend([float(weight)] * len(ds))

        num_samples = data_cfg.get("samples_per_epoch") or len(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    train_cfg = cfg["training"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_batch,
        pin_memory=True,
    )
    return loader, datasets


def build_eval_loader(cfg, split):
    """構建評估數據加載器
    
    用於構建驗證集(val)或測試集(test)的數據加載器
    如果配置中沒有指定相應的CSV文件，返回None
    
    Args:
        cfg: 配置字典
        split: 數據集分割名稱 (val/test)
    
    Returns:
        評估數據加載器，如果沒有配置則返回None
    """
    data_cfg = cfg["data"]
    csv_path = data_cfg.get(f"{split}_csv")
    if not csv_path:
        return None

    label_map = data_cfg.get(f"{split}_label_map") or data_cfg.get("label_map")
    dataset = build_dataset(cfg, csv_path, label_map, data_cfg.get("domain_id", 0), split)
    train_cfg = cfg["training"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_batch,
        pin_memory=True,
    )
    return loader


def build_optimizer(model, cfg, domain_classifier=None):
    """構建優化器
    
    為模型的不同參數組設置不同的學習率：
    - 編碼器(backbone)參數
    - 分類頭(head)參數
    - 可選的領域分類器參數（用於領域適應）
    
    Args:
        model: 情感分類模型
        cfg: 配置字典
        domain_classifier: 領域分類器，用於領域適應（可選）
    
    Returns:
        構建完成的AdamW優化器
    """
    train_cfg = cfg["training"]
    encoder_params, head_params = model.get_param_groups()
    lr_backbone = float(train_cfg.get("lr_backbone", float(train_cfg.get("lr", 1e-4))))
    lr_head = float(train_cfg.get("lr_head", lr_backbone))
    param_groups = [
        {"params": encoder_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    if domain_classifier is not None:
        param_groups.append(
            {
                "params": domain_classifier.parameters(),
                "lr": float(train_cfg.get("lr_domain", lr_head)),
            }
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    return optimizer


def build_scheduler(optimizer, cfg):
    """構建學習率調度器
    
    根據配置選擇合適的學習率調度器
    目前支持的調度器：
    - cosine: CosineAnnealingLR
    - none: 不使用調度器，返回None
    
    Args:
        optimizer: 優化器對象
        cfg: 配置字典
    
    Returns:
        學習率調度器對象，如果不使用調度器則返回None
    """
    train_cfg = cfg["training"]
    if train_cfg.get("scheduler", "none") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    return None


def save_history(path, history):
    """Save metric history to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def main(default_config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, required=default_config is None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))

    output_dir = Path(cfg.get("output_dir", "outputs"))
    checkpoint_dir = Path(cfg.get("checkpoint_dir", output_dir / "checkpoints"))

    logger = setup_logging(output_dir / "logs", cfg.get("stage", "train"))
    logger.info("Config: %s", cfg_path)

    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.info("CUDA not available, fallback to CPU")
        device = "cpu"
    device = torch.device(device)

    train_loader, train_datasets = build_train_loader(cfg)
    val_loader = build_eval_loader(cfg, "val")
    test_loader = build_eval_loader(cfg, "test")

    data_cfg = cfg["data"]
    model = EmotionModel(
        n_mels=data_cfg["n_mels"],
        num_classes=len(cfg["labels"]),
        **cfg["model"],
    ).to(device)

    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        label_ids = []
        for dataset in train_datasets:
            label_ids.extend(dataset.get_label_ids())
        class_weights = compute_class_weights(label_ids, len(cfg["labels"])).to(device)

    criterion = build_criterion(
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
        class_weights=class_weights,
    )

    domain_classifier = None
    domain_criterion = None
    if cfg["training"].get("domain_adaptation", False):
        domain_classifier = DomainClassifier(
            input_dim=model.embedding_dim,
            hidden_dim=cfg["training"].get("domain_hidden", 128),
            num_domains=cfg["training"].get("num_domains", 2),
        ).to(device)
        domain_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg, domain_classifier=domain_classifier)
    scheduler = build_scheduler(optimizer, cfg)

    metric_key = cfg["training"].get("metric_key", "macro_f1")
    best_metric = -1.0
    train_history = []
    val_history = []

    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        logger.info("Epoch %s/%s", epoch, epochs)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            len(cfg["labels"]),
            logger,
            log_every=cfg["training"].get("log_every", 20),
            max_grad_norm=cfg["training"].get("max_grad_norm"),
            domain_classifier=domain_classifier,
            domain_lambda=cfg["training"].get("domain_lambda", 0.0),
            domain_grl=cfg["training"].get("domain_grl", 1.0),
            domain_criterion=domain_criterion,
        )
        logger.info("Train metrics: %s", train_metrics)
        train_history.append(
            {
                "epoch": epoch,
                "loss": train_metrics.get("loss"),
                "accuracy": train_metrics.get("accuracy"),
            }
        )
        save_history(output_dir / "metrics" / "train_history.json", train_history)

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                len(cfg["labels"]),
                domain_classifier=domain_classifier,
                domain_criterion=domain_criterion,
            )
            logger.info("Val metrics: %s", val_metrics)
            save_metrics(output_dir / "metrics" / "val_metrics.json", val_metrics)
            val_history.append(
                {
                    "epoch": epoch,
                    "loss": val_metrics.get("loss"),
                    "accuracy": val_metrics.get("accuracy"),
                }
            )
            save_history(output_dir / "metrics" / "val_history.json", val_history)

            if val_metrics.get(metric_key, 0.0) > best_metric:
                best_metric = val_metrics.get(metric_key, 0.0)
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    cfg["labels"],
                    cfg,
                    domain_classifier=domain_classifier,
                )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            train_metrics,
            cfg["labels"],
            cfg,
            domain_classifier=domain_classifier,
        )

        if scheduler:
            scheduler.step()

    if cfg["training"].get("eval_test", False) and test_loader is not None:
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            len(cfg["labels"]),
            domain_classifier=domain_classifier,
            domain_criterion=domain_criterion,
        )
        logger.info("Test metrics: %s", test_metrics)
        save_metrics(output_dir / "metrics" / "test_metrics.json", test_metrics)


if __name__ == "__main__":
    main()
