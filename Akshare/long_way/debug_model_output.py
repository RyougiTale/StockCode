#!/usr/bin/env python3
"""
诊断模型输出问题的调试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from model_3d import create_3d_model
from data_utils import get_all_samples, calculate_features, resample_to_period
from rolling_scaler import RollingWindowScaler
from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from logger_config import setup_logging, get_logger

# 初始化日志
setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

def debug_model_predictions(stock_code='002415', model_path=None, num_samples=50):
    """调试模型预测输出"""
    
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "enhanced_pretraining", "best_loss_top_1.pth")
    
    logger.info(f"调试模型输出 - 股票: {stock_code}")
    logger.info(f"模型路径: {model_path}")
    
    # 加载模型
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # 获取数据
    logger.info("获取股票数据...")
    all_samples, scalers = get_all_samples([stock_code])
    
    if not all_samples:
        logger.error("无法获取股票数据")
        return
    
    logger.info(f"获取到 {len(all_samples)} 个样本")
    
    # 随机选择一些样本进行测试
    test_samples = np.random.choice(all_samples, min(num_samples, len(all_samples)), replace=False)
    
    predictions = []
    
    logger.info("开始模型推理...")
    
    for sample in tqdm(test_samples, desc="模型推理"):
        try:
            # 准备输入数据
            daily_features = torch.from_numpy(sample['daily'].astype(np.float32)).unsqueeze(0).to(config.DEVICE)
            weekly_features = torch.from_numpy(sample['weekly'].astype(np.float32)).unsqueeze(0).to(config.DEVICE)
            monthly_features = torch.from_numpy(sample['monthly'].astype(np.float32)).unsqueeze(0).to(config.DEVICE)
            
            # 模型推理
            with torch.no_grad():
                outputs = model(daily_features, weekly_features, monthly_features)
            
            # 解析输出
            return_logits = outputs['return'].cpu().numpy()[0]
            sharpe_logits = outputs['sharpe'].cpu().numpy()[0] 
            drawdown_logits = outputs['drawdown'].cpu().numpy()[0]
            
            # 转换为概率（注意：模型输出是log_softmax）
            return_probs = np.exp(return_logits)
            sharpe_probs = np.exp(sharpe_logits)
            drawdown_probs = np.exp(drawdown_logits)
            
            # 计算期望值
            relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            expected_return = np.sum(return_probs * relative_centers)
            expected_sharpe = np.sum(sharpe_probs * relative_centers)  
            expected_drawdown = np.sum(drawdown_probs * relative_centers)
            
            predictions.append({
                'date': sample['date'],
                'return_probs': return_probs,
                'sharpe_probs': sharpe_probs,
                'drawdown_probs': drawdown_probs,
                'expected_return': expected_return,
                'expected_sharpe': expected_sharpe,
                'expected_drawdown': expected_drawdown,
                'return_logits': return_logits,
                'sharpe_logits': sharpe_logits,
                'drawdown_logits': drawdown_logits
            })
            
        except Exception as e:
            logger.error(f"样本推理失败: {e}")
            continue
    
    if not predictions:
        logger.error("没有成功的预测结果")
        return
    
    logger.info("分析预测结果...")
    
    # 分析结果
    df_results = pd.DataFrame(predictions)
    
    print("\n" + "="*80)
    print("模型输出诊断报告")
    print("="*80)
    
    # 1. 概率分布统计
    print("\n概率分布统计:")
    for dim in ['return', 'sharpe', 'drawdown']:
        probs_array = np.array([pred[f'{dim}_probs'] for pred in predictions])
        
        print(f"\n{dim.upper()} 维度:")
        print(f"  概率和范围: [{probs_array.sum(axis=1).min():.4f}, {probs_array.sum(axis=1).max():.4f}]")
        print(f"  平均概率分布: {probs_array.mean(axis=0)}")
        print(f"  概率分布标准差: {probs_array.std(axis=0)}")
        
        # 检查是否所有预测都相同
        unique_predictions = len(np.unique(probs_array.round(4), axis=0))
        print(f"  唯一预测数量: {unique_predictions}/{len(predictions)}")
        
    # 2. 期望值统计
    print(f"\n期望值统计:")
    print(f"  Return期望值: 均值={df_results['expected_return'].mean():.4f}, 标准差={df_results['expected_return'].std():.4f}")
    print(f"  Sharpe期望值: 均值={df_results['expected_sharpe'].mean():.4f}, 标准差={df_results['expected_sharpe'].std():.4f}")
    print(f"  Drawdown期望值: 均值={df_results['expected_drawdown'].mean():.4f}, 标准差={df_results['expected_drawdown'].std():.4f}")
    
    # 3. Logits分析 
    print(f"\nLogits统计:")
    logits_stats = {}
    for dim in ['return', 'sharpe', 'drawdown']:
        logits_array = np.array([pred[f'{dim}_logits'] for pred in predictions])
        logits_stats[dim] = {
            'mean': logits_array.mean(axis=0),
            'std': logits_array.std(axis=0),
            'min': logits_array.min(axis=0),
            'max': logits_array.max(axis=0)
        }
        
        print(f"  {dim.upper()} logits均值: {logits_stats[dim]['mean']}")
        print(f"  {dim.upper()} logits标准差: {logits_stats[dim]['std']}")
    
    # 4. 稳定性测试 - 重复预测同一样本
    print(f"\n稳定性测试:")
    if predictions:
        test_sample = test_samples[0]  # 取第一个样本
        
        # 准备输入
        daily_features = torch.from_numpy(test_sample['daily'].astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        weekly_features = torch.from_numpy(test_sample['weekly'].astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        monthly_features = torch.from_numpy(test_sample['monthly'].astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        
        # 多次预测同一样本
        repeated_preds = []
        for i in range(5):
            with torch.no_grad():
                outputs = model(daily_features, weekly_features, monthly_features)
            return_probs = np.exp(outputs['return'].cpu().numpy()[0])
            repeated_preds.append(return_probs)
        
        repeated_array = np.array(repeated_preds)
        print(f"  同一样本5次预测结果:")
        for i, pred in enumerate(repeated_preds):
            print(f"    第{i+1}次: {pred}")
        print(f"  标准差: {repeated_array.std(axis=0)}")
        print(f"  最大差异: {repeated_array.max(axis=0) - repeated_array.min(axis=0)}")
    
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)

if __name__ == "__main__":
    debug_model_predictions()