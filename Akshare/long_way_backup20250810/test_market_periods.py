#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试市场时期切分功能
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加父目录到路径
parent_dir = os.path.dirname(__file__)
sys.path.insert(0, parent_dir)

import config
from dataset_3d import split_samples_by_market_periods, create_3d_datasets_with_distribution
from logger_config import get_logger, setup_logging

# 初始化日志
setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

def create_test_samples():
    """创建测试样本数据"""
    logger.info("创建测试样本数据...")
    
    samples = []
    base_date = datetime.now() - timedelta(days=365 * 12)  # 从12年前开始
    
    # 创建跨越不同时期的样本
    for i in range(1000):
        sample_date = base_date + timedelta(days=i * 4)  # 每4天一个样本
        
        sample = {
            'daily': np.random.randn(60, 12).astype(np.float32),
            'weekly': np.random.randn(52, 12).astype(np.float32), 
            'monthly': np.random.randn(24, 12).astype(np.float32),
            'future_prices': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 25)),
            'date': pd.Timestamp(sample_date),
            'stock_code': f'TEST{i%5:03d}'  # 5只测试股票
        }
        samples.append(sample)
    
    # 按时间排序
    samples.sort(key=lambda x: x['date'])
    logger.info(f"创建了 {len(samples)} 个测试样本")
    logger.info(f"时间范围: {samples[0]['date'].strftime('%Y-%m-%d')} 到 {samples[-1]['date'].strftime('%Y-%m-%d')}")
    
    return samples

def test_market_period_split():
    """测试市场时期切分功能"""
    logger.info("=== 测试市场时期切分功能 ===")
    
    # 创建测试样本
    test_samples = create_test_samples()
    
    # 测试时期切分
    logger.info("\n1. 测试时期切分功能:")
    period_samples = split_samples_by_market_periods(test_samples)
    
    total_samples = 0
    for period_name, samples in period_samples.items():
        total_samples += len(samples)
        if samples:
            start_date = samples[0]['date'].strftime('%Y-%m-%d')
            end_date = samples[-1]['date'].strftime('%Y-%m-%d')
            logger.info(f"  {period_name}: {len(samples)} 样本 ({start_date} ~ {end_date})")
    
    logger.info(f"  总样本数验证: {total_samples} == {len(test_samples)} ? {total_samples == len(test_samples)}")
    
    # 测试数据集创建
    logger.info("\n2. 测试数据集创建功能:")
    try:
        train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
            test_samples,
            train_ratio=0.7,
            val_ratio=0.15,
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        
        logger.info(f"  训练集大小: {len(train_dataset)}")
        logger.info(f"  验证集大小: {len(val_dataset)}")
        logger.info(f"  测试集大小: {len(test_dataset)}")
        logger.info(f"  股票分布数量: {len(stock_distributions)}")
        
        # 测试一个样本
        sample = train_dataset[0]
        logger.info(f"  样本形状验证:")
        logger.info(f"    Daily: {sample['daily'].shape}")
        logger.info(f"    Weekly: {sample['weekly'].shape}")
        logger.info(f"    Monthly: {sample['monthly'].shape}")
        logger.info(f"    3D标签: {list(sample['labels_3d'].keys())}")
        
        logger.info("[成功] 数据集创建功能正常")
        
    except Exception as e:
        logger.error(f"[失败] 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_config_changes():
    """测试配置参数的影响"""
    logger.info("\n3. 测试配置参数影响:")
    
    # 保存原配置
    original_config = config.MARKET_PERIOD_CONFIG.copy()
    
    try:
        # 测试禁用时期切分
        config.MARKET_PERIOD_CONFIG["enable_period_split"] = False
        test_samples = create_test_samples()[:100]  # 少量样本
        
        period_samples = split_samples_by_market_periods(test_samples)
        logger.info(f"  禁用切分后时期数: {len(period_samples)}")
        
        # 测试最小样本数限制
        config.MARKET_PERIOD_CONFIG["enable_period_split"] = True
        config.MARKET_PERIOD_CONFIG["min_samples_per_period"] = 500  # 设置很高的最小值
        
        period_samples = split_samples_by_market_periods(test_samples)
        logger.info(f"  高最小样本数限制后时期数: {len(period_samples)}")
        
        logger.info("✓ 配置参数测试正常")
        
    finally:
        # 恢复原配置
        config.MARKET_PERIOD_CONFIG.update(original_config)

if __name__ == '__main__':
    logger.info("开始市场时期切分功能测试...")
    
    success = test_market_period_split()
    if success:
        test_config_changes()
        logger.info("\n=== 所有测试通过 ===")
    else:
        logger.error("\n=== 测试失败 ===")
        sys.exit(1)