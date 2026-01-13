import csv
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from .augment import apply_augmentations
from .label_mapping import LabelMapper

# 支持的音訊文件格式
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


# 輔助函數：解析音訊路徑
def _resolve_audio_path(path_value, root_dir=None):
    """
    解析音訊文件路徑，相對路徑會轉換為絕對路徑
    
    Args:
        path_value (str): 音訊路徑字符串
        root_dir (str, optional): 根目錄路徑
        
    Returns:
        Path: 解析後的音訊文件路徑
        
    Raises:
        ValueError: 如果音訊路徑缺失
    """
    if path_value is None:
        raise ValueError("音訊清單中缺失路徑信息")
    path = Path(path_value)
    if not path.is_absolute() and root_dir:
        path = Path(root_dir) / path
    return path


# 輔助函數：讀取音訊清單
def _read_manifest(csv_path):
    """
    從CSV文件讀取音訊清單信息
    
    Args:
        csv_path (str): CSV文件路徑
        
    Returns:
        list: 音訊清單數據行列表
        
    Raises:
        FileNotFoundError: 如果CSV文件不存在
        ValueError: 如果CSV文件為空
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"音訊清單文件不存在: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"音訊清單為空: {csv_path}")
    return rows


# 輔助函數：獲取字段值
def _get_field(row, candidates):
    """
    從多個候選字段名中獲取第一個非空值
    
    Args:
        row (dict): 數據行字典
        candidates (list): 字段名候選列表
        
    Returns:
        str: 字段值（如果所有候選字段都為空則返回None）
    """
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return None


class EmotionDataset(Dataset):
    """
    情感識別音訊數據集類
    """
    
    def __init__(
        self,
        csv_path,
        label_list,
        label_map_path=None,
        sample_rate=16000,
        n_mels=80,
        n_fft=None,
        hop_length=None,
        max_duration=6.0,
        min_duration=0.5,
        mode="train",
        augment=None,
        root_dir=None,
        domain_id=None,
        drop_unknown=True,
    ):
        """
        初始化情感識別數據集
        
        Args:
            csv_path (str): 音訊清單CSV文件路徑
            label_list (list): 標籤列表
            label_map_path (str, optional): 標籤映射文件路徑
            sample_rate (int, optional): 取樣率，預設16000Hz
            n_mels (int, optional): Mel濾波器數量，預設80
            n_fft (int, optional): FFT窗口大小
            hop_length (int, optional): 跳動步長
            max_duration (float, optional): 最大音訊長度（秒）
            min_duration (float, optional): 最小音訊長度（秒）
            mode (str, optional): 數據集模式（"train"或"test"）
            augment (dict, optional): 數據增強配置
            root_dir (str, optional): 音訊文件根目錄
            domain_id (int, optional): 領域ID
            drop_unknown (bool, optional): 是否丟棄未知標籤的樣本
            
        Raises:
            ValueError: 如果未找到可用的樣本
        """
        self.csv_path = Path(csv_path)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_samples = int(max_duration * sample_rate) if max_duration else None
        self.min_samples = int(min_duration * sample_rate) if min_duration else None
        self.mode = mode
        self.augment = augment if mode == "train" else None  # 僅訓練模式使用數據增強
        self.root_dir = root_dir
        self.drop_unknown = drop_unknown
        self.label_list = list(label_list)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}

        # 加載標籤映射表
        self.label_map = LabelMapper.load_label_map(label_map_path) if label_map_path else {}

        # 加載音訊清單數據
        self.items = []
        rows = _read_manifest(self.csv_path)
        for row in rows:
            # 獲取音訊路徑（支持多種字段名）
            path_value = _get_field(row, ["path", "filepath", "audio", "wav", "file"])
            # 獲取標籤值（支持多種字段名）
            label_value = _get_field(row, ["label", "emotion", "y"])
            speaker = _get_field(row, ["speaker", "spk", "speaker_id"]) or ""
            domain = _get_field(row, ["domain", "domain_id"]) or domain_id

            if label_value is None or path_value is None:
                continue

            # 映射標籤到統一格式
            mapped_label = LabelMapper.map_label(label_value, self.label_map)
            if mapped_label not in self.label_to_id:
                if drop_unknown:
                    continue
                raise ValueError(f"未知標籤 '{mapped_label}' 出現在 {self.csv_path}")

            # 解析音訊文件路徑
            resolved = _resolve_audio_path(path_value, root_dir)
            if resolved.suffix.lower() not in AUDIO_EXTS:
                continue

            # 添加到數據集項目
            self.items.append(
                {
                    "path": str(resolved),
                    "label": self.label_to_id[mapped_label],
                    "speaker": speaker,
                    "domain": int(domain) if domain is not None else 0,
                }
            )

        if not self.items:
            raise ValueError(f"在 {self.csv_path} 中未找到可用樣本")

        # 初始化Mel頻譜轉換器
        mel_kwargs = {"sample_rate": sample_rate, "n_mels": n_mels}
        if n_fft:
            mel_kwargs["n_fft"] = n_fft
        if hop_length:
            mel_kwargs["hop_length"] = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(**mel_kwargs)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        """
        返回數據集的樣本數量
        
        Returns:
            int: 數據集樣本數
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        獲取指定索引的樣本數據
        
        Args:
            idx (int): 樣本索引
            
        Returns:
            dict: 包含特徵、長度、標籤、領域和路徑的樣本數據
        """
        item = self.items[idx]
        
        # 加載音訊文件
        waveform, sr = torchaudio.load(item["path"])
        # 將多通道轉換為單通道（取平均值）
        waveform = waveform.mean(dim=0, keepdim=True)
        # 轉換取樣率
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # 應用數據增強（僅訓練模式）
        if self.augment:
            waveform = apply_augmentations(waveform, self.sample_rate, self.augment)

        # 記錄原始音頻長度（用於計算真實Mel幀長）
        original_samples = waveform.shape[-1]
        
        # 裁剪或填充音訊到指定長度
        waveform = self._trim_or_pad(waveform)

        # 提取Mel頻譜特徵
        features = self.to_db(self.mel(waveform))
        # 調整特徵維度：(mel_bins, time) → (time, mel_bins)
        features = features.squeeze(0).transpose(0, 1)
        
        # 計算真實Mel幀長（排除padding）
        hop_length = self.hop_length if self.hop_length else self.n_fft // 4 if self.n_fft else 512
        if self.max_samples:
            max_frames = self.max_samples // hop_length + 1
            original_frames = min(original_samples // hop_length + 1, max_frames)
        else:
            original_frames = original_samples // hop_length + 1
        
        # 應用per-utterance CMVN（均值和方差歸一化）
        features = self._apply_cmvn(features, original_frames)
        
        # 截斷特徵以匹配真實長度，防止 collate_batch 發生維度不匹配錯誤
        features = features[:original_frames, :]
        
        length = original_frames

        return {
            "features": features,      # Mel頻譜特徵
            "length": length,          # 真實特徵序列長度（排除padding）
            "label": item["label"],   # 標籤ID
            "domain": item["domain"], # 領域ID
            "path": item["path"],     # 音訊文件路徑
        }

    def _trim_or_pad(self, waveform):
        """
        裁剪或填充音訊到指定的最小/最大長度
        
        Args:
            waveform (torch.Tensor): 音訊波形數據
            
        Returns:
            torch.Tensor: 處理後的音訊波形數據
        """
        num_samples = waveform.shape[-1]
        
        # 確保音訊長度不低於最小長度
        if self.min_samples and num_samples < self.min_samples:
            pad_amount = self.min_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            num_samples = waveform.shape[-1]

        # 確保音訊長度不超過最大長度
        if self.max_samples and num_samples > self.max_samples:
            if self.mode == "train":
                # 訓練模式：隨機裁剪
                start = torch.randint(0, num_samples - self.max_samples + 1, (1,)).item()
            else:
                # 測試模式：居中裁剪
                start = max(0, (num_samples - self.max_samples) // 2)
            waveform = waveform[:, start : start + self.max_samples]
            num_samples = waveform.shape[-1]

        # 最終填充到最大長度（如果需要）
        if self.max_samples and num_samples < self.max_samples:
            pad_amount = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def get_label_ids(self):
        """
        獲取所有樣本的標籤ID列表
        
        Returns:
            list: 標籤ID列表
        """
        return [item["label"] for item in self.items]

    def _apply_cmvn(self, features, valid_length):
        """
        應用per-utterance CMVN（均值和方差歸一化）
        
        Args:
            features: Mel頻譜特徵，形狀為 (seq_length, n_mels)
            valid_length: 有效序列長度（排除padding）
            
        Returns:
            歸一化後的特徵
        """
        if valid_length <= 0:
            return features
        
        # 只對有效幀進行歸一化
        valid_features = features[:valid_length, :]
        
        # 計算均值和標準差
        mean = valid_features.mean(dim=0, keepdim=True)
        std = valid_features.std(dim=0, keepdim=True)
        
        # 避免除以零
        std = std.clamp(min=1e-5)
        
        # 歸一化
        valid_features = (valid_features - mean) / std
        
        # 更新特徵
        features[:valid_length, :] = valid_features
        
        return features


# 數據批處理函數
def collate_batch(batch):
    """
    將多個樣本組合成一個批次
    
    Args:
        batch (list): 樣本列表
        
    Returns:
        dict: 批次數據字典
    """
    # 計算批次中的最大序列長度
    max_len = max(sample["length"] for sample in batch)
    feat_dim = batch[0]["features"].shape[-1]

    # 初始化批次張量
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    domains = torch.zeros(len(batch), dtype=torch.long)
    paths = []

    # 填充數據到批次中
    for idx, sample in enumerate(batch):
        length = sample["length"]
        features[idx, :length] = sample["features"]  # 填充特徵（後面會自動填充0）
        lengths[idx] = length
        labels[idx] = sample["label"]
        domains[idx] = sample["domain"]
        paths.append(sample["path"])

    return {
        "features": features,  # 批次特徵數據 (batch_size, max_len, feat_dim)
        "lengths": lengths,    # 每個樣本的實際長度 (batch_size)
        "labels": labels,      # 批次標籤 (batch_size)
        "domains": domains,    # 批次領域ID (batch_size)
        "paths": paths,        # 批次音訊路徑 (batch_size)
    }
