#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析最优的中心点设置
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from . import config
from .data_utils import get_all_samples
from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from .logger_config import get_logger

logger = get_logger(__name__)

def analyze_optimal_centers():
    """分析数据分布并推荐最优中心点"""
    
    print("="*80)
    print("分析最优中心点设置")
    print("="*80)
    
    # 获取所有样本
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    print(f"\n总样本数: {len(all_samples)}")
    
    # 创建标签生成器来计算指标
    label_generator = ImprovedThreeDimensionalLabelGenerator(
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=0.01,
        use_relative_metrics=True
    )
    
    # 收集所有指标值
    all_returns = []
    all_sharpes = []
    all_drawdowns = []
    
    for sample in all_samples:
        if 'future_prices' not in sample:
            continue
        
        future_prices = pd.Series(sample['future_prices'])
        metrics = label_generator.calculate_future_metrics(future_prices)
        
        if metrics:
            all_returns.append(metrics['total_return'])
            all_sharpes.append(metrics['sharpe_ratio'])
            all_drawdowns.append(metrics['max_drawdown'])
    
    print(f"\n有效样本数: {len(all_returns)}")
    
    # 分析每个指标的分布
    metrics_data = {
        'total_return': np.array(all_returns),
        'sharpe_ratio': np.array(all_sharpes),
        'max_drawdown': np.array(all_drawdowns)
    }
    
    recommended_centers = {}
    
    for metric_name, values in metrics_data.items():
        print(f"\n{'='*40}")
        print(f"{metric_name} 分布分析")
        print(f"{'='*40}")
        
        # 基本统计
        print(f"样本数: {len(values)}")
        print(f"均值: {np.mean(values):.4f}")
        print(f"标准差: {np.std(values):.4f}")
        print(f"最小值: {np.min(values):.4f}")
        print(f"最大值: {np.max(values):.4f}")
        
        # 分位数
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        print("\n分位数分布:")
        for p in percentiles:
            val = np.percentile(values, p)
            print(f"  {p:3d}%: {val:8.4f}")
        
        # 推荐的中心点（基于分位数）
        # 方案1：均匀分位数（10%, 30%, 50%, 70%, 90%）
        centers_uniform = [
            np.percentile(values, 10),
            np.percentile(values, 30),
            np.percentile(values, 50),
            np.percentile(values, 70),
            np.percentile(values, 90)
        ]
        
        # 方案2：考虑极值（5%, 25%, 50%, 75%, 95%）
        centers_extreme = [
            np.percentile(values, 5),
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            np.percentile(values, 95)
        ]
        
        # 方案3：基于聚类（K-means）
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(values.reshape(-1, 1))
        centers_kmeans = sorted(kmeans.cluster_centers_.flatten())
        
        print(f"\n推荐的中心点方案:")
        print(f"  均匀分位数: {[f'{c:.4f}' for c in centers_uniform]}")
        print(f"  考虑极值:   {[f'{c:.4f}' for c in centers_extreme]}")
        print(f"  K-means聚类: {[f'{c:.4f}' for c in centers_kmeans]}")
        
        # 当前使用的中心点（自适应后的）
        if hasattr(label_generator, 'relative_calculator'):
            current_centers = label_generator.relative_calculator.get_adaptive_centers(
                config.STOCK_CODES[0], metric_name
            )
            print(f"  当前中心点: {[f'{c:.4f}' for c in current_centers]}")
        
        # 推荐使用均匀分位数方案
        recommended_centers[metric_name] = centers_uniform
        
        # 可视化
        plt.figure(figsize=(12, 4))
        
        # 子图1：直方图和中心点
        plt.subplot(1, 2, 1)
        plt.hist(values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # 标记推荐的中心点
        for c in centers_uniform:
            plt.axvline(c, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel(metric_name)
        plt.ylabel('频次')
        plt.title(f'{metric_name} 分布和推荐中心点')
        plt.grid(True, alpha=0.3)
        
        # 子图2：累积分布
        plt.subplot(1, 2, 2)
        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cumulative, linewidth=2)
        
        # 标记中心点对应的累积概率
        for c in centers_uniform:
            cum_prob = np.searchsorted(sorted_values, c) / len(sorted_values)
            plt.plot(c, cum_prob, 'ro', markersize=8)
            plt.text(c, cum_prob - 0.05, f'{c:.3f}', ha='center', fontsize=8)
        
        plt.xlabel(metric_name)
        plt.ylabel('累积概率')
        plt.title('累积分布函数')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = f"{config.MODEL_DIR}/center_analysis_{metric_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")
        plt.show()
    
    # 输出建议的配置
    print("\n" + "="*80)
    print("建议的中心点配置（基于数据分布）")
    print("="*80)
    
    print("\n在 improved_label_generator.py 中修改 baseline_centers:")
    print("```python")
    print("baseline_centers = {")
    for metric_name, centers in recommended_centers.items():
        centers_str = ', '.join([f'{c:.4f}' for c in centers])
        print(f"    '{metric_name}': [{centers_str}],")
    print("}")
    print("```")
    
    # 测试新中心点的效果
    print("\n" + "="*80)
    print("测试新中心点的标签分布")
    print("="*80)
    
    # 模拟使用新中心点
    for metric_name, values in metrics_data.items():
        centers = recommended_centers[metric_name]
        
        # 计算每个样本属于哪个类别（最近的中心点）
        class_assignments = []
        for v in values:
            distances = [abs(v - c) for c in centers]
            class_assignments.append(np.argmin(distances))
        
        class_counts = np.bincount(class_assignments, minlength=5)
        class_probs = class_counts / len(class_assignments)
        
        print(f"\n{metric_name}:")
        print(f"  类别分布: {class_counts}")
        print(f"  类别概率: {[f'{p:.3f}' for p in class_probs]}")
        
        # 检查是否平衡
        min_prob = min(class_probs)
        max_prob = max(class_probs)
        balance_ratio = min_prob / max_prob if max_prob > 0 else 0
        
        if balance_ratio < 0.1:
            print(f"  ⚠️ 警告：类别不平衡严重（比率: {balance_ratio:.3f}）")
        else:
            print(f"  √ 类别平衡较好（比率: {balance_ratio:.3f}）")

if __name__ == "__main__":
    analyze_optimal_centers()