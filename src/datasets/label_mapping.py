import json
from pathlib import Path


class LabelMapper:
    """
    標籤映射工具類，用於將不同數據集的原始標籤映射到統一格式
    """
    
    @staticmethod
    def load_label_map(path):
        """
        從JSON文件加載標籤映射表
        
        Args:
            path (str): 標籤映射文件路徑
            
        Returns:
            dict: 標籤映射字典
            
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if not path:
            return {}
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"標籤映射文件不存在: {path}")
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)

    @staticmethod
    def map_label(raw_label, label_map):
        """
        使用映射表將原始標籤映射到目標標籤
        
        Args:
            raw_label (str): 原始標籤
            label_map (dict): 標籤映射字典
            
        Returns:
            str: 映射後的標籤（如果無映射則返回原始標籤）
        """
        if label_map is None:
            return raw_label
        return label_map.get(raw_label, raw_label)
