#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析3D模型预测准确性的诊断工具
帮助理解为什么loss低但预测值看起来不准
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import pearsonr, spearmanr

# 使用相对导入（需要用 python -m long_way.analyze_prediction_accuracy 运行）
from . import config
from .draw_3d_long_term import predict_3d_long_term
from .logger_config import get_logger

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__)

def calculate_distribution_similarity(pred_values, actual_values, n_bins=20):
    """
    计算预测值和实际值分布的相似度
    """
    from scipy.stats import wasserstein_distance, ks_2samp
    from scipy.spatial.distance import jensenshannon
    
    # 移除NaN
    mask = ~(np.isnan(pred_values) | np.isnan(actual_values))
    pred_clean = pred_values[mask]
    actual_clean = actual_values[mask]
    
    if len(pred_clean) < 10:
        return {}
    
    # 计算直方图
    min_val = min(pred_clean.min(), actual_clean.min())
    max_val = max(pred_clean.max(), actual_clean.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    pred_hist, _ = np.histogram(pred_clean, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual_clean, bins=bins, density=True)
    
    # 归一化为概率分布
    pred_hist = pred_hist / pred_hist.sum()
    actual_hist = actual_hist / actual_hist.sum()
    
    # 计算多种分布相似度指标
    metrics = {
        'wasserstein': wasserstein_distance(pred_clean, actual_clean),
        'js_divergence': jensenshannon(pred_hist, actual_hist),
        'ks_statistic': ks_2samp(pred_clean, actual_clean)[0],
        'ks_pvalue': ks_2samp(pred_clean, actual_clean)[1],
        'mean_diff': np.mean(pred_clean) - np.mean(actual_clean),
        'std_ratio': np.std(pred_clean) / np.std(actual_clean) if np.std(actual_clean) > 0 else np.inf,
    }
    
    return metrics, bins, pred_hist, actual_hist

def analyze_prediction_accuracy(stock_code="002415", model_path=None, years=1):
    """
    深入分析预测准确性
    
    Args:
        stock_code: 股票代码
        model_path: 模型文件路径，如果为None则使用默认的best_loss_top_1.pth
        years: 分析的年数
    """
    logger.info(f"开始分析 {stock_code} 的预测准确性...")
    
    # 如果没有指定模型路径，使用默认路径
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "best_loss_top_1.pth")
    
    # 获取预测数据
    df = predict_3d_long_term(stock_code, model_path, years)
    
    # 只分析有实际数据的部分
    df_analysis = df[df['actual_return'].notna()].copy()
    
    if df_analysis.empty:
        logger.error("没有可分析的数据")
        return
    
    # 创建分析图表 - 增加到4行以包含分布分析
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    # 1. 收益率分析
    metrics = ['return', 'sharpe', 'drawdown']
    metric_names = ['收益率', '夏普比率', '最大回撤']
    
    for row, (metric, name) in enumerate(zip(metrics, metric_names)):
        actual_col = f'actual_{metric}'
        
        # 1.1 散点图：预测vs实际
        ax = axes[row, 0]
        for pred_type, color, label in [
            ('full', 'blue', '全概率'),
            ('top3', 'green', 'Top-3'),
            ('top2', 'purple', 'Top-2'),
            ('top1', 'red', 'Top-1')
        ]:
            pred_col = f'pred_{metric}_{pred_type}'
            if pred_col in df_analysis.columns:
                ax.scatter(df_analysis[actual_col], df_analysis[pred_col], 
                          alpha=0.5, s=10, label=label, color=color)
        
        # 添加对角线
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        ax.set_xlabel(f'实际{name}')
        ax.set_ylabel(f'预测{name}')
        ax.set_title(f'{name} - 预测vs实际散点图')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 1.2 误差分布直方图
        ax = axes[row, 1]
        errors = {}
        for pred_type in ['full', 'top3', 'top2', 'top1']:
            pred_col = f'pred_{metric}_{pred_type}'
            if pred_col in df_analysis.columns:
                error = df_analysis[pred_col] - df_analysis[actual_col]
                errors[pred_type] = error
                ax.hist(error, bins=30, alpha=0.5, label=pred_type.replace('_', '-').title())
        
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel(f'{name}预测误差')
        ax.set_ylabel('频次')
        ax.set_title(f'{name} - 误差分布')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 1.3 时间序列相关性
        ax = axes[row, 2]
        # 计算滚动相关系数
        window = 20  # 20天滚动窗口
        for pred_type, color in [('full', 'blue'), ('top1', 'red')]:
            pred_col = f'pred_{metric}_{pred_type}'
            if pred_col in df_analysis.columns:
                rolling_corr = []
                dates = []
                for i in range(window, len(df_analysis)):
                    subset = df_analysis.iloc[i-window:i]
                    corr = subset[actual_col].corr(subset[pred_col])
                    rolling_corr.append(corr)
                    dates.append(subset.iloc[-1]['date'])
                
                ax.plot(dates, rolling_corr, label=pred_type.replace('_', '-').title(), 
                       color=color, linewidth=1.5)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='green', linestyle='--', alpha=0.3)
        ax.set_xlabel('日期')
        ax.set_ylabel('相关系数')
        ax.set_title(f'{name} - 20日滚动相关系数')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # 4. 分布分析（第4行）
    for col, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[3, col]
        actual_col = f'actual_{metric}'
        pred_col = f'pred_{metric}_full'
        
        if pred_col in df_analysis.columns:
            # 计算分布相似度
            dist_metrics, bins, pred_hist, actual_hist = calculate_distribution_similarity(
                df_analysis[pred_col].values, 
                df_analysis[actual_col].values
            )
            
            # 绘制分布对比
            bin_centers = (bins[:-1] + bins[1:]) / 2
            width = bins[1] - bins[0]
            
            ax.bar(bin_centers - width/4, actual_hist, width/2, 
                   label='实际分布', alpha=0.7, color='blue')
            ax.bar(bin_centers + width/4, pred_hist, width/2, 
                   label='预测分布', alpha=0.7, color='red')
            
            ax.set_xlabel(name)
            ax.set_ylabel('概率密度')
            ax.set_title(f'{name} - 分布对比\nJS散度={dist_metrics["js_divergence"]:.3f}, KS={dist_metrics["ks_statistic"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{stock_code} - 预测准确性深度分析', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    save_path = os.path.join(config.MODEL_DIR, f"accuracy_analysis_{stock_code}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"分析图表已保存: {save_path}")
    
    # 打印统计分析
    print("\n" + "="*80)
    print(f"{stock_code} 预测准确性统计分析")
    print("="*80)
    
    for metric, name in zip(metrics, metric_names):
        print(f"\n【{name}】")
        actual_col = f'actual_{metric}'
        
        for pred_type in ['full', 'top3', 'top2', 'top1']:
            pred_col = f'pred_{metric}_{pred_type}'
            if pred_col in df_analysis.columns:
                # 计算各种指标
                mae = np.mean(np.abs(df_analysis[pred_col] - df_analysis[actual_col]))
                rmse = np.sqrt(np.mean((df_analysis[pred_col] - df_analysis[actual_col])**2))
                pearson_corr, p_value = pearsonr(df_analysis[actual_col], df_analysis[pred_col])
                spearman_corr, _ = spearmanr(df_analysis[actual_col], df_analysis[pred_col])
                
                # 方向准确率（预测涨跌方向是否正确）
                if metric == 'return':
                    actual_direction = np.sign(df_analysis[actual_col])
                    pred_direction = np.sign(df_analysis[pred_col])
                    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                else:
                    direction_accuracy = 0
                
                print(f"\n  {pred_type.upper()}预测:")
                print(f"    MAE: {mae:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    Pearson相关: {pearson_corr:.4f} (p={p_value:.4f})")
                print(f"    Spearman相关: {spearman_corr:.4f}")
                if metric == 'return':
                    print(f"    方向准确率: {direction_accuracy:.1f}%")
        
        # 添加分布相似度分析
        print(f"\n  分布相似度分析:")
        pred_col_full = f'pred_{metric}_full'
        if pred_col_full in df_analysis.columns and actual_col in df_analysis.columns:
            dist_metrics, _, _, _ = calculate_distribution_similarity(
                df_analysis[pred_col_full].values,
                df_analysis[actual_col].values
            )
            if dist_metrics:
                print(f"    Wasserstein距离: {dist_metrics['wasserstein']:.4f}")
                print(f"    JS散度: {dist_metrics['js_divergence']:.4f}")
                print(f"    KS统计量: {dist_metrics['ks_statistic']:.4f} (p值={dist_metrics['ks_pvalue']:.4f})")
                print(f"    均值差异: {dist_metrics['mean_diff']:.4f}")
                print(f"    标准差比率: {dist_metrics['std_ratio']:.4f}")
    
    # 分析为什么loss低但预测看起来不准
    print("\n" + "="*80)
    print("为什么Loss低但预测看起来不准？")
    print("="*80)
    
    print("\n1. 软标签vs硬预测的差异:")
    print("   - 模型训练时优化的是5类概率分布的交叉熵")
    print("   - 但可视化时我们将概率分布转换为单一数值")
    print("   - 这个转换过程会丢失信息并引入误差")
    
    print("\n2. 相对指标的尺度变换:")
    print("   - 相对指标将不同股票映射到统一的[0,1]空间")
    print("   - 但反向映射时，小的相对误差可能对应大的绝对误差")
    print("   - 特别是对于波动大的股票")
    
    print("\n3. 分布预测vs点预测:")
    print("   - 模型实际预测的是'未来表现的概率分布'")
    print("   - 即使分布预测准确，期望值也可能偏离实际值")
    print("   - 这是概率预测的固有特性")
    
    # 显示实际的概率分布示例
    print("\n4. 概率分布示例（最后10个预测）:")
    print("-"*60)
    
    # 这里需要重新运行模型获取概率分布，暂时跳过
    
    plt.show()
    
    return df_analysis

def analyze_relative_distributions(stock_code="002415", model_path=None, years=1):
    """
    分析相对位置的分布（因为模型实际上是在相对空间中训练的）
    """
    logger.info(f"分析 {stock_code} 的相对位置分布...")
    
    # 如果没有指定模型路径，使用默认路径
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "best_loss_top_1.pth")
    
    # 获取预测数据
    df = predict_3d_long_term(stock_code, model_path, years)
    df_analysis = df[df['actual_return'].notna()].copy()
    
    if df_analysis.empty:
        logger.error("没有可分析的数据")
        return
    
    # 转换为相对位置（百分位数）
    from scipy.stats import percentileofscore
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    metrics = ['return', 'sharpe', 'drawdown']
    metric_names = ['收益率', '夏普比率', '最大回撤']
    
    for col, (metric, name) in enumerate(zip(metrics, metric_names)):
        actual_col = f'actual_{metric}'
        pred_col = f'pred_{metric}_full'
        
        if pred_col in df_analysis.columns:
            # 转换为相对位置（0-1）
            actual_values = df_analysis[actual_col].values
            pred_values = df_analysis[pred_col].values
            
            # 计算每个值在整体分布中的百分位
            actual_relative = np.array([percentileofscore(actual_values, v) / 100 for v in actual_values])
            pred_relative = np.array([percentileofscore(pred_values, v) / 100 for v in pred_values])
            
            # 上图：相对位置的散点图
            ax = axes[0, col]
            ax.scatter(actual_relative, pred_relative, alpha=0.5, s=10)
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
            ax.set_xlabel(f'实际相对位置')
            ax.set_ylabel(f'预测相对位置')
            ax.set_title(f'{name} - 相对位置对比\n相关系数={np.corrcoef(actual_relative, pred_relative)[0,1]:.3f}')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # 下图：相对位置的分布
            ax = axes[1, col]
            ax.hist(actual_relative, bins=20, alpha=0.5, label='实际', color='blue', density=True)
            ax.hist(pred_relative, bins=20, alpha=0.5, label='预测', color='red', density=True)
            ax.set_xlabel('相对位置')
            ax.set_ylabel('密度')
            ax.set_title(f'{name} - 相对位置分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{stock_code} - 相对位置分布分析（这是模型真正学习的空间）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(config.MODEL_DIR, f"relative_distribution_{stock_code}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"相对分布图表已保存: {save_path}")
    
    plt.show()
    
    return df_analysis

def compare_different_aggregations(stock_code="002415", model_path=None):
    """
    比较不同聚合方式的效果
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "best_loss_top_1.pth")
    df = predict_3d_long_term(stock_code, model_path, years=1)
    df_analysis = df[df['actual_return'].notna()].copy()
    
    if df_analysis.empty:
        return
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, name) in enumerate(zip(['return', 'sharpe', 'drawdown'], 
                                           ['收益率', '夏普比率', '最大回撤'])):
        ax = axes[i]
        actual_col = f'actual_{metric}'
        
        # 计算不同聚合方式的准确性
        accuracies = []
        for pred_type in ['full', 'top3', 'top2', 'top1']:
            pred_col = f'pred_{metric}_{pred_type}'
            if pred_col in df_analysis.columns:
                corr = df_analysis[actual_col].corr(df_analysis[pred_col])
                accuracies.append((pred_type, corr))
        
        # 绘制条形图
        pred_types = [a[0] for a in accuracies]
        corrs = [a[1] for a in accuracies]
        colors = ['blue', 'green', 'purple', 'red']
        
        bars = ax.bar(pred_types, corrs, color=colors[:len(pred_types)], alpha=0.7)
        ax.set_ylabel('相关系数')
        ax.set_title(f'{name} - 不同聚合方式的准确性')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, corr in zip(bars, corrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'{stock_code} - 不同概率聚合方式的比较', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行分析
    import sys
    if len(sys.argv) > 1:
        stock_code = sys.argv[1]
    else:
        stock_code = "002415"  # 默认股票
    
    print(f"\n分析股票: {stock_code}")
    print("\n1. 详细准确性分析（包含分布对比）")
    df = analyze_prediction_accuracy(stock_code, years=1)
    
    print("\n2. 相对位置分布分析（模型训练空间）")
    analyze_relative_distributions(stock_code, years=1)
    
    print("\n3. 比较不同聚合方式")
    compare_different_aggregations(stock_code)