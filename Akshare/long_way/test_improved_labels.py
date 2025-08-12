#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的标签生成策略
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
from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from dataset_3d import Market3DClassificationDataset
from logger_config import get_logger, setup_logging

setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

def test_improved_labels():
    """测试改进的标签生成"""
    logger.info("=== 测试改进的标签生成策略 ===")
    
    # 创建测试数据
    test_samples = []
    np.random.seed(42)
    
    for i in range(1000):
        # 模拟不同类型的future_prices（更真实的分布）
        if i < 200:  # 大涨 15±5%
            base_return = np.random.normal(0.15, 0.05)
        elif i < 400:  # 小涨 5±3%
            base_return = np.random.normal(0.05, 0.03)
        elif i < 600:  # 平盘 0±2%
            base_return = np.random.normal(0.0, 0.02)
        elif i < 800:  # 小跌 -5±3%
            base_return = np.random.normal(-0.05, 0.03)
        else:  # 大跌 -15±5%
            base_return = np.random.normal(-0.15, 0.05)
            
        # 生成更真实的价格序列
        returns = np.random.normal(base_return/20, 0.01, 25)
        prices = 100 * np.cumprod(1 + returns)
        
        sample = {
            'daily': np.random.randn(60, 12).astype(np.float32),
            'weekly': np.random.randn(52, 12).astype(np.float32),
            'monthly': np.random.randn(24, 12).astype(np.float32),
            'future_prices': prices,
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'stock_code': f'TEST{i%5:03d}'  # 5只测试股票
        }
        test_samples.append(sample)
    
    logger.info(f"创建测试样本: {len(test_samples)} 个")
    
    # 测试不同温度参数
    temperatures = [0.05, 0.1, 0.2, 0.5]
    
    for temp in temperatures:
        logger.info(f"\\n--- 温度参数 {temp} (改进的相对化指标) ---")
        
        # 创建改进的数据集
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
        
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            return_labels.append(sample['labels_3d']['return'].numpy())
            sharpe_labels.append(sample['labels_3d']['sharpe'].numpy())
            drawdown_labels.append(sample['labels_3d']['drawdown'].numpy())
        
        return_labels = np.array(return_labels)
        sharpe_labels = np.array(sharpe_labels)
        drawdown_labels = np.array(drawdown_labels)
        
        # 计算统计指标
        def analyze_labels(labels, name):
            # 熵（衡量软标签的"软度"）
            eps = 1e-8
            entropy = -np.sum(labels * np.log(labels + eps), axis=1).mean()
            
            # 最大概率（衡量"硬度"）
            max_prob = labels.max(axis=1).mean()
            
            # 标准差（衡量分布的离散程度）
            std = labels.std(axis=1).mean()
            
            # 类别分布
            class_counts = []
            for class_idx in range(5):
                class_counts.append((labels.argmax(axis=1) == class_idx).sum())
            
            logger.info(f"  {name}标签:")
            logger.info(f"    平均熵: {entropy:.4f} (越高越'软')")
            logger.info(f"    平均最大概率: {max_prob:.4f} (越低越'软')")
            logger.info(f"    平均标准差: {std:.4f}")
            logger.info(f"    类别分布: {class_counts}")
        
        analyze_labels(return_labels, "Return")
        analyze_labels(sharpe_labels, "Sharpe")
        analyze_labels(drawdown_labels, "Drawdown")
        
        # 检查是否所有标签都相同（之前的问题）
        unique_return = len(np.unique(return_labels.round(4), axis=0))
        unique_sharpe = len(np.unique(sharpe_labels.round(4), axis=0))
        unique_drawdown = len(np.unique(drawdown_labels.round(4), axis=0))
        
        logger.info(f"  标签唯一性检查:")
        logger.info(f"    Return唯一标签数: {unique_return}")
        logger.info(f"    Sharpe唯一标签数: {unique_sharpe}")
        logger.info(f"    Drawdown唯一标签数: {unique_drawdown}")
        
        if unique_return == 1 and unique_sharpe == 1 and unique_drawdown == 1:
            logger.warning("  ⚠️  所有标签仍然相同！")
        else:
            logger.info("  ✅ 标签有区分度！")

def compare_old_vs_new():
    """对比旧的和新的标签生成策略"""
    logger.info("\\n=== 对比旧的和新的标签生成策略 ===")
    
    # 创建相同的测试数据
    test_samples = []
    np.random.seed(123)  # 固定种子确保可比较性
    
    for i in range(500):
        if i < 100:
            base_return = 0.15  # 大涨
        elif i < 200:
            base_return = 0.05  # 小涨
        elif i < 300:
            base_return = 0.0   # 平盘
        elif i < 400:
            base_return = -0.05 # 小跌
        else:
            base_return = -0.15 # 大跌
            
        returns = np.random.normal(base_return/20, 0.01, 25)
        prices = 100 * np.cumprod(1 + returns)
        
        sample = {
            'daily': np.random.randn(60, 12).astype(np.float32),
            'weekly': np.random.randn(52, 12).astype(np.float32),
            'monthly': np.random.randn(24, 12).astype(np.float32),
            'future_prices': prices,
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'stock_code': f'TEST{i%3:03d}'
        }
        test_samples.append(sample)
    
    # 测试新策略
    new_dataset = Market3DClassificationDataset(
        test_samples, 
        temperature=0.1,
        use_relative_metrics=True
    )
    
    # 分析结果
    new_return_labels = []
    for i in range(min(50, len(new_dataset))):
        sample = new_dataset[i]
        new_return_labels.append(sample['labels_3d']['return'].numpy())
    
    new_return_labels = np.array(new_return_labels)
    unique_new = len(np.unique(new_return_labels.round(4), axis=0))
    
    logger.info(f"新策略结果:")
    logger.info(f"  唯一Return标签数: {unique_new}")
    logger.info(f"  平均熵: {-np.sum(new_return_labels * np.log(new_return_labels + 1e-8), axis=1).mean():.4f}")
    
    if unique_new > 5:
        logger.info("✅ 新策略生成了有区分度的标签！")
    else:
        logger.warning("⚠️  新策略仍有问题")

if __name__ == '__main__':
    logger.info("开始测试改进的标签生成策略...")
    
    try:
        test_improved_labels()
        compare_old_vs_new()
        
        logger.info("\\n=== 测试完成 ===")
        logger.info("如果看到'标签有区分度'的信息，说明修复成功！")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()