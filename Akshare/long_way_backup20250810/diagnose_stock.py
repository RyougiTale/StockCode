import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import config
from .data_utils import get_all_samples
from .dataset import create_soft_label

def analyze_stock_data():
    """深入分析股票数据，找出NaN损失的根本原因"""
    print("=== 股票数据深度诊断 ===")
    print(f"分析股票代码: {config.STOCK_CODES}")
    
    # 1. 获取原始数据
    print("\n--- 1. 获取和分析原始数据 ---")
    all_samples, scalers = get_all_samples(config.STOCK_CODES)
    
    if not all_samples:
        print("ERROR: 无法获取样本数据")
        return
    
    print(f"总样本数: {len(all_samples)}")
    
    # 2. 分析标签分布
    print("\n--- 2. 标签分布分析 ---")
    labels = [s['label'] for s in all_samples]
    labels_array = np.array(labels)
    
    print(f"标签统计:")
    print(f"  数量: {len(labels_array)}")
    print(f"  均值: {labels_array.mean():.6f}")
    print(f"  标准差: {labels_array.std():.6f}")
    print(f"  最小值: {labels_array.min():.6f}")
    print(f"  最大值: {labels_array.max():.6f}")
    print(f"  中位数: {np.median(labels_array):.6f}")
    
    # 分析极端值
    percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
    print(f"\n百分位数分析:")
    for p in percentiles:
        val = np.percentile(labels_array, p)
        print(f"  {p:2d}%: {val:8.4f}")
    
    # 找出最极端的值
    extreme_threshold = 0.5  # 50%
    extreme_indices = np.where(np.abs(labels_array) > extreme_threshold)[0]
    print(f"\n极端值分析 (绝对值 > {extreme_threshold*100}%):")
    print(f"  极端值数量: {len(extreme_indices)} ({len(extreme_indices)/len(labels_array)*100:.2f}%)")
    
    if len(extreme_indices) > 0:
        print(f"  极端值范围: [{labels_array[extreme_indices].min():.4f}, {labels_array[extreme_indices].max():.4f}]")
        print(f"  前10个极端值:")
        for i, idx in enumerate(extreme_indices[:10]):
            sample = all_samples[idx]
            print(f"    {i+1:2d}. 日期: {sample['date'].strftime('%Y-%m-%d')}, 回报率: {sample['label']:8.4f}")
    
    # 3. 分析软标签生成
    print("\n--- 3. 软标签生成分析 ---")
    class_centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"]
    temperature = config.SOFT_LABEL_CONFIG["TEMPERATURE"]
    
    print(f"类别中心: {class_centers.numpy()}")
    print(f"温度参数: {temperature}")
    
    # 测试几个极端值的软标签生成
    test_returns = [labels_array.min(), labels_array.max(), 0.0, 0.1, -0.1]
    if len(extreme_indices) > 0:
        test_returns.extend([labels_array[extreme_indices[0]], labels_array[extreme_indices[-1]]])
    
    print(f"\n软标签生成测试:")
    problematic_returns = []
    
    for ret in test_returns:
        try:
            true_return = torch.tensor(ret, dtype=torch.float32)
            soft_label = create_soft_label(true_return, class_centers, temperature)
            
            min_prob = soft_label.min().item()
            max_prob = soft_label.max().item()
            entropy = -(soft_label * torch.log(soft_label + 1e-8)).sum().item()
            
            print(f"  回报率 {ret:8.4f}: 软标签 {soft_label.numpy()}")
            print(f"    -> 最小概率: {min_prob:.2e}, 最大概率: {max_prob:.4f}, 熵: {entropy:.4f}")
            
            # 检查是否有问题
            if min_prob < 1e-10 or torch.isnan(soft_label).any():
                problematic_returns.append(ret)
                print(f"    -> ⚠️  问题标签!")
                
        except Exception as e:
            print(f"  回报率 {ret:8.4f}: ERROR - {e}")
            problematic_returns.append(ret)
    
    # 4. 分析输入特征
    print("\n--- 4. 输入特征分析 ---")
    
    # 随机选择几个样本检查
    sample_indices = np.random.choice(len(all_samples), min(5, len(all_samples)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = all_samples[idx]
        print(f"\n样本 {i+1} (日期: {sample['date'].strftime('%Y-%m-%d')}):")
        
        for timeframe, data in [('daily', sample['daily']), ('weekly', sample['weekly']), ('monthly', sample['monthly'])]:
            data_flat = data.flatten()
            nan_count = np.isnan(data_flat).sum()
            inf_count = np.isinf(data_flat).sum()
            
            print(f"  {timeframe:7s}: shape={data.shape}, nan={nan_count}, inf={inf_count}")
            print(f"           range=[{data_flat.min():.6f}, {data_flat.max():.6f}]")
            
            if nan_count > 0 or inf_count > 0:
                print(f"           ⚠️  发现异常值!")
    
    # 5. 总结和建议
    print("\n--- 5. 诊断总结 ---")
    
    issues_found = []
    
    if len(problematic_returns) > 0:
        issues_found.append(f"发现 {len(problematic_returns)} 个有问题的回报率值")
    
    if len(extreme_indices) > len(all_samples) * 0.05:  # 超过5%的极端值
        issues_found.append(f"极端值比例过高: {len(extreme_indices)/len(all_samples)*100:.1f}%")
    
    if temperature < 0.01:
        issues_found.append(f"温度参数可能过小: {temperature}")
    
    if len(issues_found) > 0:
        print("发现的问题:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        
        print("\n建议的解决方案:")
        if len(problematic_returns) > 0:
            print("  - 在软标签生成中增加数值稳定性检查")
        if len(extreme_indices) > len(all_samples) * 0.05:
            print("  - 考虑对极端值进行截断或特殊处理")
        if temperature < 0.01:
            print("  - 考虑适当增加温度参数")
    else:
        print("未发现明显问题，可能需要更深入的分析")
    
    return {
        'total_samples': len(all_samples),
        'extreme_count': len(extreme_indices),
        'problematic_returns': problematic_returns,
        'labels_stats': {
            'mean': labels_array.mean(),
            'std': labels_array.std(),
            'min': labels_array.min(),
            'max': labels_array.max()
        }
    }

if __name__ == '__main__':
    analyze_stock_data()