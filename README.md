# Paipai Speech Emotion

本專案包含兩部分：

1. **前端錄音 + 情緒展示**（`demo/`）
2. **兩階段訓練 Pipeline**（Stage 1: CASIA，Stage 2: CHEAVD/自然情緒）

---

## 训练快速开始（仅 Stage 1，ChEAVD 還沒拿到）

> 如果没有 CHEAVD，直接跑 Stage 1 就能用。

### 0) 安装依赖

```bash
pip install -r requirements.txt
```

### 1) 确认 CASIA 位置

当前数据放在：

```
data/raw/CASIA/CASIA中文语音情感识别/CASIA database/
```

如果你的路径不同，后面的 `--input_dir` 改成你的实际路径。

### 2) 生成 manifest

```bash
python scripts/prepare_manifest.py \
  --mode folder \
  --input_dir "data/raw/CASIA/CASIA中文语音情感识别/CASIA database" \
  --output_csv data/splits/casia/all.csv \
  --domain_id 0 \
  --relative_to data/raw
```

### 3) 切分 train/val/test（speaker-disjoint）

```bash
python scripts/split_manifest.py \
  --input_csv data/splits/casia/all.csv \
  --output_dir data/splits/casia \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

已生成的分割文件在：`data/splits/casia/`（`train.csv` / `val.csv` / `test.csv`），如需重新切分可重复执行上面命令。

### 4) 开始训练 Stage 1

```bash
python scripts/train_stage1.py
```

- 没有 GPU 会自动 fallback 到 CPU（很慢）。
- Windows 如果卡住，改 `configs/stage1_casia.yaml` 的 `training.num_workers: 0`。

---

## Stage 2（可选）

只有当你拿到 CHEAVD/自然情绪数据时才需要：

1. 放入 `data/raw/cheavd/`
2. 用 `scripts/prepare_manifest.py` 生成 manifest
3. 更新 `configs/stage2_cheavd.yaml` 的路径
4. 运行：`python scripts/train_stage2.py`

---

## 推理测试（单文件）

```bash
python scripts/predict.py \
  --checkpoint checkpoints/stage1/best.pt \
  --audio /path/to/test.wav \
  --device cpu
```

---

## 训练曲线可视化

训练结束后会在 `outputs/metrics/` 生成 `train_history.json` 和 `val_history.json`。运行：

```bash
python scripts/plot_metrics.py
```

会在 `outputs/plots/` 生成 `train_metrics.png` / `val_metrics.png` / `test_metrics.png`。

---

## 前端使用

1. 打开 `demo/index.html`
2. 点击 **Start recording** / **Stop recording**
3. 点击 **Analyze** 调用后端模型

后端默认 API：`POST /api/analyze`，`multipart/form-data`，字段 `file`。

---

## 文档入口

- 训练完整说明：`docs/pipeline.md`
- 数据集存放说明：`data/raw/README.md`

---

## 专案结构

```
paipai_speech_emo/
├── demo/                 # 前端录音展示
├── configs/              # Stage 1 / Stage 2 設定檔
├── src/                  # dataset / model / training code
├── scripts/              # 訓練、評估、資料處理入口
├── data/                 # label map / split CSV / raw data
├── docs/                 # Pipeline 文件
├── checkpoints/          # 模型輸出
├── outputs/              # metrics / logs
└── README.md
```
