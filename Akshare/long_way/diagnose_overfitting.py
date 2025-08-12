#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度诊断过拟合问题
检查数据预处理、标签生成、数据泄露等所有可能问题
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
long_way_dir = os.path.join(parent_dir, 'long_way')
sys.path.insert(0, long_way_dir)

import config
from dataset_3d import create_3d_datasets_with_distribution, Market3DClassificationDataset
from data_utils import get_all_samples
from label_3d_generator import ThreeDimensionalLabelGenerator
from logger_config import get_logger, setup_logging

setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

def analyze_label_distribution():
    """分析标签分布的合理性"""
    logger.info("=== 分析3D标签分布 ===")
    
    # 创建小规模测试数据
    test_samples = []
    np.random.seed(42)
    
    for i in range(1000):
        # 模拟不同类型的future_prices
        if i < 200:  # 大涨
            base_return = np.random.normal(0.15, 0.05)
        elif i < 400:  # 小涨
            base_return = np.random.normal(0.05, 0.03)
        elif i < 600:  # 平盘
            base_return = np.random.normal(0.0, 0.02)
        elif i < 800:  # 小跌
            base_return = np.random.normal(-0.05, 0.03)
        else:  # 大跌
            base_return = np.random.normal(-0.15, 0.05)
            
        # 生成价格序列
        prices = 100 * np.cumprod(1 + np.random.normal(base_return/20, 0.01, 25))
        
        sample = {
            'daily': np.random.randn(60, 12).astype(np.float32),
            'weekly': np.random.randn(52, 12).astype(np.float32),
            'monthly': np.random.randn(24, 12).astype(np.float32),
            'future_prices': prices,
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'stock_code': f'TEST{i%5:03d}'
        }
        test_samples.append(sample)
    
    logger.info(f"创建测试样本: {len(test_samples)} 个")
    
    # 测试不同温度参数的效果
    temperatures = [0.002, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    for temp in temperatures:
        logger.info(f"\\n--- 温度参数 {temp} ---")
        
        # 创建数据集
        dataset = Market3DClassificationDataset(
            test_samples, 
            look_forward_days=20, 
            temperature=temp,
            use_relative_metrics=True
        )
        
        # 分析标签分布
        return_labels = []
        sharpe_labels = []
        drawdown_labels = []
        
        for i in range(min(100, len(dataset))):  # 只分析前100个
            sample = dataset[i]
            return_labels.append(sample['labels_3d']['return'].numpy())
            sharpe_labels.append(sample['labels_3d']['sharpe'].numpy())
            drawdown_labels.append(sample['labels_3d']['drawdown'].numpy())
        
        return_labels = np.array(return_labels)
        sharpe_labels = np.array(sharpe_labels)
        drawdown_labels = np.array(drawdown_labels)
        
        # 计算熵（衡量软标签的"软度"）
        def calculate_entropy(probs):
            eps = 1e-8
            return -np.sum(probs * np.log(probs + eps), axis=1).mean()
        
        return_entropy = calculate_entropy(return_labels)
        sharpe_entropy = calculate_entropy(sharpe_labels)
        drawdown_entropy = calculate_entropy(drawdown_labels)
        
        logger.info(f"  Return标签熵: {return_entropy:.4f}")
        logger.info(f"  Sharpe标签熵: {sharpe_entropy:.4f}")
        logger.info(f"  Drawdown标签熵: {drawdown_entropy:.4f}")
        
        # 计算最大概率（衡量"硬度"）
        return_max_prob = return_labels.max(axis=1).mean()
        sharpe_max_prob = sharpe_labels.max(axis=1).mean()
        drawdown_max_prob = drawdown_labels.max(axis=1).mean()
        
        logger.info(f"  Return平均最大概率: {return_max_prob:.4f}")
        logger.info(f"  Sharpe平均最大概率: {sharpe_max_prob:.4f}")
        logger.info(f"  Drawdown平均最大概率: {drawdown_max_prob:.4f}")

def analyze_normalization_effects():
    """分析归一化对数据的影响"""
    logger.info("\\n=== 分析归一化效果 ===")
    
    # 获取真实数据的小样本
    all_samples, scalers = get_all_samples(['002415', '600519', '000001'])  # 只取3只股票
    if not all_samples:
        logger.error("无法获取样本数据")
        return
    
    logger.info(f"获取样本数: {len(all_samples)}")
    
    # 分析归一化前后的特征分布
    sample_features = []
    for sample in all_samples[:100]:  # 只分析前100个样本
        daily_features = sample['daily']
        sample_features.append(daily_features)
    
    sample_features = np.array(sample_features)  # shape: (samples, time_steps, features)
    
    logger.info(f"特征矩阵形状: {sample_features.shape}")
    
    # 分析每个特征的分布
    feature_names = config.FEATURE_COLUMNS['daily']
    for i, feature_name in enumerate(feature_names[:5]):  # 只分析前5个特征
        if i < sample_features.shape[2]:
            feature_data = sample_features[:, :, i].flatten()
            
            logger.info(f"\\n特征 {feature_name}:")
            logger.info(f"  均值: {np.mean(feature_data):.6f}")
            logger.info(f"  标准差: {np.std(feature_data):.6f}")
            logger.info(f"  最小值: {np.min(feature_data):.6f}")
            logger.info(f"  最大值: {np.max(feature_data):.6f}")
            logger.info(f"  NaN数量: {np.isnan(feature_data).sum()}")
            logger.info(f"  Inf数量: {np.isinf(feature_data).sum()}")

def analyze_data_leakage():
    """分析潜在的数据泄露问题"""
    logger.info("\\n=== 分析数据泄露风险 ===")
    
    # 获取数据并创建数据集
    all_samples, _ = get_all_samples(['002415', '600519'])  # 只取2只股票测试
    if len(all_samples) < 100:
        logger.error("样本数量不足")
        return
    
    all_samples = all_samples[:500]  # 限制样本数量加快分析
    
    logger.info(f"分析样本数: {len(all_samples)}")
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
        all_samples,
        train_ratio=0.7,
        val_ratio=0.15,
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True
    )
    
    logger.info(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
    
    # 分析时间重叠
    train_dates = []
    val_dates = []
    test_dates = []
    
    for i in range(min(100, len(train_dataset))):
        # 这里需要从原始samples中获取日期信息，但dataset可能没有直接访问
        pass  # 暂时跳过，需要修改dataset类来支持这种分析
    
    logger.info("数据泄露分析需要进一步的数据集修改来支持")

def analyze_relative_metrics():
    """分析相对化指标是否引入过多噪声"""
    logger.info("\\n=== 分析相对化指标 ===")
    
    # 创建标签生成器
    generator_relative = ThreeDimensionalLabelGenerator(
        temperature=0.1,
        use_relative_metrics=True
    )
    
    generator_absolute = ThreeDimensionalLabelGenerator(
        temperature=0.1, 
        use_relative_metrics=False
    )
    
    # 模拟相同的指标数据
    test_metrics = [
        {'total_return': 0.1, 'sharpe_ratio': 1.5, 'max_drawdown': -0.05},
        {'total_return': -0.05, 'sharpe_ratio': -0.5, 'max_drawdown': -0.15},
        {'total_return': 0.2, 'sharpe_ratio': 2.0, 'max_drawdown': -0.02},
    ]
    
    # 需要先构建分布才能使用相对化指标
    # 这个分析需要真实的样本数据来构建分布
    logger.info("相对化指标分析需要真实样本数据来构建股票分布")

if __name__ == '__main__':
    logger.info("开始深度诊断过拟合问题...")
    
    try:
        analyze_label_distribution()
        analyze_normalization_effects()
        analyze_data_leakage()
        analyze_relative_metrics()
        
        logger.info("\\n=== 诊断完成 ===")
        logger.info("关键发现:")
        logger.info("1. 检查温度参数对软标签'硬度'的影响")
        logger.info("2. 检查归一化后数据的分布是否合理")
        logger.info("3. 需要进一步分析数据泄露和相对化指标")
        
    except Exception as e:
        logger.error(f"诊断过程中出现错误: {e}")
        import traceback
        traceback.print_exc()