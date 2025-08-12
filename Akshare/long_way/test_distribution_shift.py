#!/usr/bin/env python3
"""
测试训练股票与新股票的分布差异
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from data_utils import get_all_samples
import config
from scipy import stats
import matplotlib.pyplot as plt

def analyze_stock_distributions(trained_stocks, new_stock):
    """分析训练股票与新股票的分布差异"""
    
    print(f"分析股票分布差异...")
    print(f"训练股票样本: {trained_stocks[:5]}")
    print(f"新股票: {new_stock}")
    
    # 获取训练股票的样本
    trained_samples = []
    for stock in trained_stocks[:10]:  # 只取前10只避免太慢
        samples, _ = get_all_samples([stock])
        if samples:
            trained_samples.extend(samples[:100])  # 每只股票取100个样本
            print(f"  {stock}: 获取 {len(samples)} 个样本")
    
    # 获取新股票的样本
    new_samples, _ = get_all_samples([new_stock])
    if not new_samples:
        print(f"无法获取 {new_stock} 的数据")
        return
    
    print(f"\n{new_stock}: 获取 {len(new_samples)} 个样本")
    
    # 提取特征进行比较
    def extract_features(samples):
        """提取日度特征的统计信息"""
        daily_features = []
        for sample in samples[:min(100, len(samples))]:
            if 'daily' in sample:
                daily_features.append(sample['daily'].flatten())
        
        if not daily_features:
            return None
            
        features = np.array(daily_features)
        return {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0)
        }
    
    trained_stats = extract_features(trained_samples)
    new_stats = extract_features(new_samples)
    
    if trained_stats is None or new_stats is None:
        print("无法提取特征统计")
        return
    
    print("\n特征分布对比:")
    print(f"训练股票特征:")
    print(f"  均值范围: [{trained_stats['mean'].min():.4f}, {trained_stats['mean'].max():.4f}]")
    print(f"  标准差范围: [{trained_stats['std'].min():.4f}, {trained_stats['std'].max():.4f}]")
    
    print(f"\n新股票({new_stock})特征:")
    print(f"  均值范围: [{new_stats['mean'].min():.4f}, {new_stats['mean'].max():.4f}]")
    print(f"  标准差范围: [{new_stats['std'].min():.4f}, {new_stats['std'].max():.4f}]")
    
    # 计算分布距离
    # KL散度（需要正值化）
    trained_mean_dist = trained_stats['mean'] - trained_stats['mean'].min() + 1
    new_mean_dist = new_stats['mean'] - new_stats['mean'].min() + 1
    
    # 归一化
    trained_mean_dist = trained_mean_dist / trained_mean_dist.sum()
    new_mean_dist = new_mean_dist / new_mean_dist.sum()
    
    kl_div = stats.entropy(new_mean_dist, trained_mean_dist)
    
    print(f"\n分布差异度量:")
    print(f"  KL散度: {kl_div:.4f}")
    print(f"  均值差异L2范数: {np.linalg.norm(trained_stats['mean'] - new_stats['mean']):.4f}")
    print(f"  标准差差异L2范数: {np.linalg.norm(trained_stats['std'] - new_stats['std']):.4f}")
    
    # 检查是否存在极端差异
    mean_ratio = np.abs(new_stats['mean'] / (trained_stats['mean'] + 1e-8))
    extreme_features = np.where((mean_ratio > 10) | (mean_ratio < 0.1))[0]
    
    if len(extreme_features) > 0:
        print(f"\n警告: 发现 {len(extreme_features)} 个特征存在极端差异（10倍以上）")
        print(f"  极端特征索引: {extreme_features[:10].tolist()}")
    
    return {
        'kl_divergence': kl_div,
        'mean_l2': np.linalg.norm(trained_stats['mean'] - new_stats['mean']),
        'std_l2': np.linalg.norm(trained_stats['std'] - new_stats['std']),
        'extreme_features': len(extreme_features)
    }

if __name__ == "__main__":
    # 使用配置中的股票
    trained_stocks = config.STOCK_CODES[:10]  # 取前10只训练股票
    new_stock = '002415'  # 测试股票
    
    results = analyze_stock_distributions(trained_stocks, new_stock)
    
    if results:
        print("\n" + "="*50)
        print("分析总结:")
        if results['kl_divergence'] > 1.0:
            print("[X] 新股票与训练股票分布差异很大")
        elif results['kl_divergence'] > 0.5:
            print("[!] 新股票与训练股票存在中等分布差异")
        else:
            print("[OK] 新股票与训练股票分布相似")
        
        if results['extreme_features'] > 0:
            print(f"[X] 存在 {results['extreme_features']} 个极端差异特征，这可能是导致预测失败的原因")
        print("="*50)