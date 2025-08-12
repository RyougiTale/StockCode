#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同温度参数对3D标签分布的影响
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# 使用相对导入
from . import config
from .data_utils import get_all_samples
from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from .logger_config import get_logger

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__)

def test_temperature_effects():
    """测试不同温度对标签分布的影响"""
    
    # 测试的温度值
    temperatures = [0.002, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    # 获取样本
    print("获取样本数据...")
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    print(f"总样本数: {len(all_samples)}")
    
    if len(all_samples) < 100:
        print("警告：样本数太少！")
        return
    
    # 为每个温度创建标签生成器并分析
    results = {}
    
    for temp in temperatures:
        print(f"\n{'='*50}")
        print(f"测试温度 = {temp}")
        print('='*50)
        
        # 创建标签生成器
        label_generator = ImprovedThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=temp,
            use_relative_metrics=True
        )
        
        # 构建股票分布
        stock_samples_dict = {}
        for sample in all_samples:
            stock_code = sample.get('stock_code', config.STOCK_CODES[0])
            if stock_code not in stock_samples_dict:
                stock_samples_dict[stock_code] = []
            stock_samples_dict[stock_code].append(sample)
        
        label_generator.fit_stock_distributions(stock_samples_dict)
        
        # 收集标签
        all_return_labels = []
        all_sharpe_labels = []
        all_drawdown_labels = []
        
        for i in range(min(200, len(all_samples))):
            sample = all_samples[i]
            if 'future_prices' not in sample:
                continue
                
            # 转换为pandas Series
            import pandas as pd
            future_prices = pd.Series(sample['future_prices'])
            metrics = label_generator.calculate_future_metrics(future_prices)
            if metrics is None:
                continue
                
            soft_labels = label_generator.create_soft_label_3d(
                metrics,
                sample.get('stock_code', config.STOCK_CODES[0])
            )
            
            all_return_labels.append(soft_labels['return'].numpy())
            all_sharpe_labels.append(soft_labels['sharpe'].numpy())
            all_drawdown_labels.append(soft_labels['drawdown'].numpy())
        
        if len(all_return_labels) == 0:
            continue
            
        all_return_labels = np.array(all_return_labels)
        all_sharpe_labels = np.array(all_sharpe_labels)
        all_drawdown_labels = np.array(all_drawdown_labels)
        
        # 分析结果
        results[temp] = {
            'return': all_return_labels,
            'sharpe': all_sharpe_labels,
            'drawdown': all_drawdown_labels
        }
        
        # 打印统计
        print(f"\n收益率标签:")
        print(f"  平均分布: {all_return_labels.mean(axis=0)}")
        print(f"  标准差: {all_return_labels.std(axis=0)}")
        print(f"  最大概率平均值: {all_return_labels.max(axis=1).mean():.3f}")
        print(f"  独特标签数: {len(np.unique(all_return_labels.round(4), axis=0))}/{len(all_return_labels)}")
        
        # 计算熵
        epsilon = 1e-8
        entropy = -np.sum(all_return_labels * np.log(all_return_labels + epsilon), axis=1).mean()
        max_entropy = np.log(5)
        print(f"  平均熵: {entropy:.3f}/{max_entropy:.3f} (比例: {entropy/max_entropy:.2%})")
        
        # 类别分布
        argmax_classes = all_return_labels.argmax(axis=1)
        class_counts = np.bincount(argmax_classes, minlength=5)
        print(f"  预测类别分布: {class_counts} ({class_counts/len(all_return_labels)*100}%)")
        
        print(f"\n夏普比率标签:")
        print(f"  平均分布: {all_sharpe_labels.mean(axis=0)}")
        print(f"  独特标签数: {len(np.unique(all_sharpe_labels.round(4), axis=0))}/{len(all_sharpe_labels)}")
        
        print(f"\n最大回撤标签:")
        print(f"  平均分布: {all_drawdown_labels.mean(axis=0)}")
        print(f"  独特标签数: {len(np.unique(all_drawdown_labels.round(4), axis=0))}/{len(all_drawdown_labels)}")
    
    # 可视化对比
    if results:
        plot_temperature_comparison(results)
    
    # 推荐最佳温度
    print("\n" + "="*50)
    print("推荐的温度设置:")
    print("="*50)
    print("基于测试结果，建议使用温度 = 0.01 到 0.05")
    print("- 太低（<0.01）：标签过于尖锐，可能过拟合")
    print("- 太高（>0.1）：标签过于平滑，失去区分度")
    
def plot_temperature_comparison(results):
    """可视化不同温度的效果"""
    temps = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 收益率标签
    for i, metric in enumerate(['return', 'sharpe', 'drawdown']):
        ax = axes[0, i]
        
        # 绘制平均最大概率
        max_probs = []
        for temp in temps:
            labels = results[temp][metric]
            max_probs.append(labels.max(axis=1).mean())
        
        ax.plot(temps, max_probs, 'o-', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('温度')
        ax.set_ylabel('平均最大概率')
        ax.set_title(f'{metric} - 最大概率 vs 温度')
        ax.grid(True, alpha=0.3)
        
        # 绘制独特标签数
        ax = axes[1, i]
        unique_counts = []
        for temp in temps:
            labels = results[temp][metric]
            unique_counts.append(len(np.unique(labels.round(4), axis=0)))
        
        ax.plot(temps, unique_counts, 'o-', linewidth=2, color='orange')
        ax.set_xscale('log')
        ax.set_xlabel('温度')
        ax.set_ylabel('独特标签数')
        ax.set_title(f'{metric} - 标签多样性 vs 温度')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('不同温度参数对3D标签分布的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(config.MODEL_DIR, "temperature_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n分析图表已保存: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    test_temperature_effects()