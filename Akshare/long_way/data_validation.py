#!/usr/bin/env python3
"""
增强的数据验证工具，防止NaN/Inf导致的训练不稳定
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
try:
    from .logger_config import get_logger
except ImportError:
    from logger_config import get_logger

logger = get_logger(__name__)

def validate_tensor_batch(tensor_dict: Dict[str, torch.Tensor], batch_idx: int = None) -> bool:
    """
    验证tensor批次数据，检测NaN/Inf
    
    Args:
        tensor_dict: 包含tensor的字典
        batch_idx: 批次索引（用于日志）
    
    Returns:
        bool: True if valid, False if contains NaN/Inf
    """
    has_invalid = False
    
    for key, tensor in tensor_dict.items():
        if torch.isnan(tensor).any():
            logger.error(f"批次 {batch_idx}: {key} 包含 NaN 值")
            has_invalid = True
            
        if torch.isinf(tensor).any():
            logger.error(f"批次 {batch_idx}: {key} 包含 Inf 值")
            has_invalid = True
            
        # 检查异常大的值
        max_val = tensor.abs().max().item()
        if max_val > 1e6:
            logger.warning(f"批次 {batch_idx}: {key} 包含异常大的值: {max_val}")
            
        # 检查零方差
        if tensor.numel() > 1:
            std_val = tensor.std().item()
            if std_val < 1e-8:
                logger.warning(f"批次 {batch_idx}: {key} 标准差过小: {std_val}")
    
    return not has_invalid

def clean_tensor_batch(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    清理tensor批次数据，替换NaN/Inf值
    
    Args:
        tensor_dict: 包含tensor的字典
    
    Returns:
        Dict[str, torch.Tensor]: 清理后的tensor字典
    """
    cleaned_dict = {}
    
    for key, tensor in tensor_dict.items():
        cleaned_tensor = tensor.clone()
        
        # 替换NaN为0
        nan_mask = torch.isnan(cleaned_tensor)
        if nan_mask.any():
            cleaned_tensor[nan_mask] = 0.0
            logger.warning(f"替换了 {nan_mask.sum().item()} 个 NaN 值在 {key}")
        
        # 替换Inf为较大但有限的值
        inf_mask = torch.isinf(cleaned_tensor)
        if inf_mask.any():
            # 替换正无穷为1e6，负无穷为-1e6
            cleaned_tensor[inf_mask] = torch.sign(cleaned_tensor[inf_mask]) * 1e6
            logger.warning(f"替换了 {inf_mask.sum().item()} 个 Inf 值在 {key}")
        
        # 裁剪异常大的值
        cleaned_tensor = torch.clamp(cleaned_tensor, -1e6, 1e6)
        
        cleaned_dict[key] = cleaned_tensor
    
    return cleaned_dict

def validate_sample_data(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    验证样本数据，移除包含NaN/Inf的样本
    
    Args:
        samples: 样本数据列表
    
    Returns:
        Tuple[List[Dict[str, Any]], int]: (清理后的样本, 移除的样本数)
    """
    valid_samples = []
    removed_count = 0
    
    for sample in samples:
        is_valid = True
        
        for key in ['daily', 'weekly', 'monthly']:
            if key in sample:
                data = sample[key]
                if isinstance(data, np.ndarray):
                    if np.isnan(data).any() or np.isinf(data).any():
                        logger.debug(f"移除包含 NaN/Inf 的样本 {key} 数据")
                        is_valid = False
                        break
                    
                    # 检查异常大的值
                    max_val = np.abs(data).max()
                    if max_val > 1e6:
                        logger.debug(f"移除包含异常值的样本 {key}: {max_val}")
                        is_valid = False
                        break
        
        # 检查标签
        if 'label' in sample and sample['label'] is not None:
            if isinstance(sample['label'], (int, float)):
                if np.isnan(sample['label']) or np.isinf(sample['label']):
                    logger.debug(f"移除包含无效标签的样本")
                    is_valid = False
        
        if is_valid:
            valid_samples.append(sample)
        else:
            removed_count += 1
    
    if removed_count > 0:
        logger.info(f"数据验证完成: 移除了 {removed_count} 个无效样本，保留了 {len(valid_samples)} 个有效样本")
    
    return valid_samples, removed_count

def check_data_distribution(samples: List[Dict[str, Any]], sample_size: int = 1000) -> Dict[str, Any]:
    """
    检查数据分布统计信息
    
    Args:
        samples: 样本数据
        sample_size: 抽样大小
    
    Returns:
        Dict[str, Any]: 分布统计信息
    """
    if len(samples) == 0:
        return {}
    
    # 随机抽样
    sample_indices = np.random.choice(len(samples), min(sample_size, len(samples)), replace=False)
    sampled_data = [samples[i] for i in sample_indices]
    
    stats = {}
    
    for data_type in ['daily', 'weekly', 'monthly']:
        if data_type in sampled_data[0]:
            all_data = np.array([sample[data_type] for sample in sampled_data])
            
            stats[data_type] = {
                'mean': np.mean(all_data),
                'std': np.std(all_data),
                'min': np.min(all_data),
                'max': np.max(all_data),
                'shape': all_data.shape
            }
    
    logger.info("数据分布统计:")
    for data_type, stat in stats.items():
        logger.info(f"  {data_type}: 均值={stat['mean']:.4f}, 标准差={stat['std']:.4f}, "
                   f"范围=[{stat['min']:.4f}, {stat['max']:.4f}], 形状={stat['shape']}")
    
    return stats