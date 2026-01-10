"""音頻數據增強模塊

提供多種音頻數據增強技術，用於訓練階段提高模型的泛化能力
包括加噪、時間偏移、速度擾動和SpecAugment等操作
"""
import random

import torch
import torchaudio


def apply_noise(waveform, noise_std):
    """向波形添加高斯噪聲

    Args:
        waveform: 輸入音頻波形，形狀為 (channels, samples)
        noise_std: 噪聲標準差，控制噪聲強度

    Returns:
        添加噪聲後的波形，形狀與輸入相同
    """
    if noise_std <= 0:
        return waveform
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise


def apply_time_shift(waveform, max_shift):
    """對波形進行時間偏移

    Args:
        waveform: 輸入音頻波形，形狀為 (channels, samples)
        max_shift: 最大偏移比例 (0~1)，實際偏移樣本數為 max_shift * 總樣本數

    Returns:
        時間偏移後的波形，形狀與輸入相同
    """
    if max_shift <= 0:
        return waveform
    shift = int(random.uniform(-max_shift, max_shift) * waveform.shape[-1])
    if shift == 0:
        return waveform
    return torch.roll(waveform, shifts=shift, dims=-1)


def apply_speed_perturb(waveform, sample_rate, speeds):
    """對波形進行速度擾動（改變播放速度）

    Args:
        waveform: 輸入音頻波形，形狀為 (channels, samples)
        sample_rate: 原始采樣率
        speeds: 速度因子列表，例如 [0.9, 1.0, 1.1]

    Returns:
        速度擾動後的波形，長度可能改變但采樣率保持不變
    """
    if not speeds:
        return waveform
    speed = random.choice(speeds)
    if speed == 1.0:
        return waveform
    new_rate = int(sample_rate * speed)
    waveform = torchaudio.functional.resample(waveform, sample_rate, new_rate)
    waveform = torchaudio.functional.resample(waveform, new_rate, sample_rate)
    return waveform


def apply_augmentations(waveform, sample_rate, config):
    """根據配置應用多種增強技術的組合

    Args:
        waveform: 輸入音頻波形，形狀為 (channels, samples)
        sample_rate: 采樣率
        config: 增強配置字典，包含以下鍵：
            - enabled: 是否啟用增強
            - speed_perturb: 速度因子列表
            - time_shift: 最大時間偏移比例
            - noise_std: 噪聲標準差

    Returns:
        應用增強後的波形，形狀與輸入相同
    """
    if not config or not config.get("enabled", False):
        return waveform
    waveform = apply_speed_perturb(
        waveform, sample_rate, config.get("speed_perturb", [])
    )
    waveform = apply_time_shift(waveform, config.get("time_shift", 0.0))
    waveform = apply_noise(waveform, config.get("noise_std", 0.0))
    return waveform


def apply_time_mask(features, num_masks, max_mask_ratio):
    """對Mel頻譜應用時間遮罩（SpecAugment）

    Args:
        features: Mel頻譜特徵，形狀為 (seq_length, n_mels)
        num_masks: 時間遮罩的數量
        max_mask_ratio: 每個遮罩的最大長度比例（相對於序列長度）

    Returns:
        應用時間遮罩後的特徵
    """
    if num_masks <= 0:
        return features
    
    seq_length = features.shape[0]
    for _ in range(num_masks):
        mask_length = int(random.uniform(0, max_mask_ratio) * seq_length)
        if mask_length == 0:
            continue
        mask_start = random.randint(0, seq_length - mask_length)
        features[mask_start:mask_start + mask_length, :] = 0
    
    return features


def apply_freq_mask(features, num_masks, max_freq_ratio):
    """對Mel頻譜應用頻率遮罩（SpecAugment）

    Args:
        features: Mel頻譜特徵，形狀為 (seq_length, n_mels)
        num_masks: 頻率遮罩的數量
        max_freq_ratio: 每個遮罩的最大寬度比例（相對於頻率維度）

    Returns:
        應用頻率遮罩後的特徵
    """
    if num_masks <= 0:
        return features
    
    n_mels = features.shape[1]
    for _ in range(num_masks):
        mask_width = int(random.uniform(0, max_freq_ratio) * n_mels)
        if mask_width == 0:
            continue
        mask_start = random.randint(0, n_mels - mask_width)
        features[:, mask_start:mask_start + mask_width] = 0
    
    return features


def apply_specaugment(features, config):
    """應用SpecAugment（時間和頻率遮罩）

    Args:
        features: Mel頻譜特徵，形狀為 (seq_length, n_mels)
        config: SpecAugment配置字典，包含以下鍵：
            - time_masks: 時間遮罩數量
            - time_mask_ratio: 時間遮罩最大長度比例
            - freq_masks: 頻率遮罩數量
            - freq_mask_ratio: 頻率遮罩最大寬度比例

    Returns:
        應用SpecAugment後的特徵
    """
    if not config or not config.get("enabled", False):
        return features
    
    features = apply_time_mask(
        features,
        config.get("time_masks", 2),
        config.get("time_mask_ratio", 0.1)
    )
    features = apply_freq_mask(
        features,
        config.get("freq_masks", 2),
        config.get("freq_mask_ratio", 0.1)
    )
    
    return features