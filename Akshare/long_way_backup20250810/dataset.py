import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from . import config
from .logger_config import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 软标签生成函数保持不变，但我们可以把它放在 Dataset 类的外面或者作为静态方法
def create_soft_label(true_return, class_centers, temperature):
    """
    根据真实的连续回报率，生成一个软标签概率分布。
    增加数值稳定性检查。
    """
    # 检查输入是否有异常值
    if torch.isnan(true_return) or torch.isinf(true_return):
        if config.DEBUG_MODE:
            logger.warning(f"无效的true_return: {true_return}")
        # 返回均匀分布作为fallback
        return torch.ones(len(class_centers)) / len(class_centers)
    
    distances = (true_return - class_centers).pow(2)
    scores = -distances / temperature
    
    # 检查scores是否有异常值
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        if config.DEBUG_MODE:
            logger.warning(f"软标签生成中出现无效scores: true_return={true_return}, distances={distances}, scores={scores}")
        return torch.ones(len(class_centers)) / len(class_centers)
    
    # 为了数值稳定性，限制scores的范围
    scores = torch.clamp(scores, min=-50, max=50)
    
    soft_label = F.softmax(scores, dim=-1)
    
    # 最终检查软标签
    if torch.isnan(soft_label).any() or torch.isinf(soft_label).any():
        if config.DEBUG_MODE:
            logger.warning(f"生成了无效的软标签: {soft_label}")
        return torch.ones(len(class_centers)) / len(class_centers)
    
    # 确保软标签不会太接近0（避免KL散度中的log(0)）
    epsilon = 1e-8
    soft_label = torch.clamp(soft_label, min=epsilon, max=1.0-epsilon)
    # 重新归一化
    soft_label = soft_label / soft_label.sum()
    
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
        logger.info("预计算所有软标签...")
        self.soft_labels = []
        invalid_count = 0
        extreme_return_count = 0
        
        for i, sample in enumerate(self.samples):
            true_return = torch.tensor(sample['label'], dtype=torch.float32)
            
            # 统计极端回报率
            if abs(sample['label']) > 0.5:  # 超过50%的回报率
                extreme_return_count += 1
                if extreme_return_count <= 5 and config.DEBUG_MODE:  # 只打印前5个
                    logger.debug(f"发现极端回报: {sample['label']:.4f} 在 {sample['date']}")
            
            # 调用函数生成软标签
            sl = create_soft_label(true_return, class_centers, temperature)
            
            # 检查生成的软标签
            if torch.isnan(sl).any() or torch.isinf(sl).any() or (sl < 1e-8).any():
                invalid_count += 1
                if invalid_count <= 5 and config.DEBUG_MODE:  # 只打印前5个无效的
                    logger.warning(f"第 {i} 个样本的软标签无效: {sl}")
            
            self.soft_labels.append(sl)
        
        logger.info(f"软标签预计算完成: {len(self.samples)} 个样本, {extreme_return_count} 个极端回报(>50%), {invalid_count} 个无效标签")
        
        if invalid_count > 0:
            logger.warning(f"发现 {invalid_count} 个无效软标签！")
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