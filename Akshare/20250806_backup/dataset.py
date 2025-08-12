import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from . import config

# 软标签生成函数保持不变，但我们可以把它放在 Dataset 类的外面或者作为静态方法
def create_soft_label(true_return, class_centers, temperature):
    """
    根据真实的连续回报率，生成一个软标签概率分布。
    """
    # 这个函数现在只处理单个数值，不需要device转换
    distances = (true_return - class_centers).pow(2)
    scores = -distances / temperature
    soft_label = F.softmax(scores, dim=-1)
    return soft_label

class MarketClassificationDataset(Dataset):
    """
    用于市场状态分类任务的PyTorch数据集（性能优化版）。
    """
    def __init__(self, samples):
        """
        Args:
            samples (list of dict): 样本列表。
        """
        self.samples = samples
        
        # --- 性能优化的核心改动 ---
        # 1. 从config加载配置
        class_centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"]
        temperature = config.SOFT_LABEL_CONFIG["TEMPERATURE"]
        
        # 2. 在初始化时，一次性计算并存储所有软标签
        print("Pre-calculating all soft labels...")
        self.soft_labels = []
        for sample in self.samples:
            true_return = torch.tensor(sample['label'], dtype=torch.float32)
            # 调用函数生成软标签
            sl = create_soft_label(true_return, class_centers, temperature)
            self.soft_labels.append(sl)
        print("Soft labels pre-calculation finished.")
        # --- 改动结束 ---

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本。
        这个方法现在变得非常快，因为它只做索引和数据类型转换。
        """
        sample = self.samples[idx]
        
        # 将输入数据转换为 torch.float32 类型
        daily_data = torch.from_numpy(sample['daily'].astype(np.float32))
        weekly_data = torch.from_numpy(sample['weekly'].astype(np.float32))
        monthly_data = torch.from_numpy(sample['monthly'].astype(np.float32))
        
        # 直接从预先计算好的列表中获取软标签
        soft_label = self.soft_labels[idx]
        
        return {
            'daily': daily_data,
            'weekly': weekly_data,
            'monthly': monthly_data,
            'label': soft_label
        }