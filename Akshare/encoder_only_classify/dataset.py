import torch
from torch.utils.data import Dataset
import numpy as np

class MarketClassificationDataset(Dataset):
    """
    用于市场状态分类任务的PyTorch数据集。
    """
    def __init__(self, samples):
        """
        Args:
            samples (list of dict): 从 data_utils.get_all_samples 返回的样本列表。
                                    每个样本是一个字典，包含 'daily', 'weekly', 'monthly', 'label'。
        """
        self.samples = samples

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本。
        将Numpy数组转换为PyTorch张量。
        """
        sample = self.samples[idx]
        
        # 将数据转换为 torch.float32 类型
        daily_data = torch.from_numpy(sample['daily'].astype(np.float32))
        weekly_data = torch.from_numpy(sample['weekly'].astype(np.float32))
        monthly_data = torch.from_numpy(sample['monthly'].astype(np.float32))
        
        # 标签是类别，通常使用 torch.long 类型
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'daily': daily_data,
            'weekly': weekly_data,
            'monthly': monthly_data,
            'label': label
        }