#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断训练问题 - 为什么模型预测失效
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# 使用相对导入
from . import config
from .dataset_3d import create_3d_datasets_with_distribution
from .data_utils import get_all_samples
from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from .logger_config import get_logger
from .model_3d import create_3d_model
from tqdm import tqdm

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__)

def diagnose_label_distribution():
    """诊断标签分布问题"""
    print("\n" + "="*80)
    print("诊断1: 检查训练标签分布")
    print("="*80)
    
    # 获取训练数据
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    
    if not all_samples:
        print("错误：无法获取训练样本")
        return
    
    print(f"总样本数: {len(all_samples)}")
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset, _ = create_3d_datasets_with_distribution(
        all_samples, 
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 分析标签分布
    def analyze_labels(dataset, name):
        print(f"\n--- {name} 标签分析 ---")
        
        all_return_labels = []
        all_sharpe_labels = []
        all_drawdown_labels = []
        
        # 收集前100个样本的标签
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            labels = sample['labels_3d']
            all_return_labels.append(labels['return'].numpy())
            all_sharpe_labels.append(labels['sharpe'].numpy())
            all_drawdown_labels.append(labels['drawdown'].numpy())
        
        all_return_labels = np.array(all_return_labels)
        all_sharpe_labels = np.array(all_sharpe_labels)
        all_drawdown_labels = np.array(all_drawdown_labels)
        
        # 计算每个类别的平均概率
        return_mean = all_return_labels.mean(axis=0)
        sharpe_mean = all_sharpe_labels.mean(axis=0)
        drawdown_mean = all_drawdown_labels.mean(axis=0)
        
        print(f"收益率标签平均分布: {return_mean}")
        print(f"夏普比率标签平均分布: {sharpe_mean}")
        print(f"最大回撤标签平均分布: {drawdown_mean}")
        
        # 计算标签的标准差（应该不要太小）
        return_std = all_return_labels.std(axis=0)
        sharpe_std = all_sharpe_labels.std(axis=0)
        drawdown_std = all_drawdown_labels.std(axis=0)
        
        print(f"收益率标签标准差: {return_std}")
        print(f"夏普比率标签标准差: {sharpe_std}")
        print(f"最大回撤标签标准差: {drawdown_std}")
        
        # 检查是否有退化（所有标签都一样）
        unique_returns = len(np.unique(all_return_labels.round(3), axis=0))
        unique_sharpes = len(np.unique(all_sharpe_labels.round(3), axis=0))
        unique_drawdowns = len(np.unique(all_drawdown_labels.round(3), axis=0))
        
        print(f"独特的收益率标签数: {unique_returns}/{len(all_return_labels)}")
        print(f"独特的夏普标签数: {unique_sharpes}/{len(all_sharpe_labels)}")
        print(f"独特的回撤标签数: {unique_drawdowns}/{len(all_drawdown_labels)}")
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 收益率标签分布
        axes[0].boxplot(all_return_labels.T)
        axes[0].set_title('收益率标签分布')
        axes[0].set_xlabel('类别')
        axes[0].set_ylabel('概率')
        
        # 夏普比率标签分布
        axes[1].boxplot(all_sharpe_labels.T)
        axes[1].set_title('夏普比率标签分布')
        axes[1].set_xlabel('类别')
        
        # 最大回撤标签分布
        axes[2].boxplot(all_drawdown_labels.T)
        axes[2].set_title('最大回撤标签分布')
        axes[2].set_xlabel('类别')
        
        plt.suptitle(f'{name} - 标签分布箱线图', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return all_return_labels, all_sharpe_labels, all_drawdown_labels
    
    train_labels = analyze_labels(train_dataset, "训练集")
    val_labels = analyze_labels(val_dataset, "验证集")
    
    return train_labels, val_labels

def diagnose_model_outputs():
    """诊断模型输出问题"""
    print("\n" + "="*80)
    print("诊断2: 检查模型输出分布")
    print("="*80)
    
    # 加载模型
    model_path = os.path.join(config.MODEL_DIR, "best_loss_top_1.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # 使用一只股票的所有样本进行测试（参考draw_3d_long_term.py）
    test_stock = config.STOCK_CODES[0]
    print(f"使用股票 {test_stock} 的所有样本进行模型输出诊断...")
    
    all_samples, scaler = get_all_samples([test_stock])
    if not all_samples:
        print(f"无法获取股票 {test_stock} 的数据")
        return
    
    print(f"股票 {test_stock} 总样本数: {len(all_samples)}")
    
    # 创建完整的数据集（用于获取完整的股票分布）
    train_dataset, val_dataset, test_dataset, _ = create_3d_datasets_with_distribution(
        all_samples,
        train_ratio=0.7,
        val_ratio=0.15,
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True
    )
    
    print(f"数据集分割: 训练{len(train_dataset)}, 验证{len(val_dataset)}, 测试{len(test_dataset)}")
    
    # 收集模型输出（使用测试集）
    all_outputs = {'return': [], 'sharpe': [], 'drawdown': []}
    sample_count = min(200, len(test_dataset))  # 使用更多样本
    
    print(f"分析 {sample_count} 个测试样本的模型输出...")
    
    with torch.no_grad():
        for i in tqdm(range(sample_count), desc="模型预测"):
            sample = test_dataset[i]
            
            # 添加batch维度
            daily = sample['daily'].unsqueeze(0).to(config.DEVICE)
            weekly = sample['weekly'].unsqueeze(0).to(config.DEVICE)
            monthly = sample['monthly'].unsqueeze(0).to(config.DEVICE)
            
            outputs = model(daily, weekly, monthly)
            
            # 转换为概率
            for key in ['return', 'sharpe', 'drawdown']:
                # 模型输出的已经是log_softmax，需要exp得到概率
                probs = torch.exp(outputs[key]).cpu().numpy()[0]
                all_outputs[key].append(probs)
    
    # 分析输出
    for key in ['return', 'sharpe', 'drawdown']:
        outputs = np.array(all_outputs[key])
        print(f"\n{key} 模型输出:")
        print(f"  平均概率分布: {outputs.mean(axis=0)}")
        print(f"  标准差: {outputs.std(axis=0)}")
        print(f"  最大概率的平均值: {outputs.max(axis=1).mean():.3f}")
        
        # 检查是否总是预测同一个类别
        predicted_classes = outputs.argmax(axis=1)
        class_counts = Counter(predicted_classes)
        print(f"  预测类别分布: {dict(class_counts)}")
        
        # 检查熵（不确定性）
        epsilon = 1e-8
        entropy = -np.sum(outputs * np.log(outputs + epsilon), axis=1).mean()
        max_entropy = np.log(5)  # 5个类别的最大熵
        print(f"  平均熵: {entropy:.3f} (最大: {max_entropy:.3f})")
        
        # 检查是否过度集中在某些类别
        class_probs = outputs.mean(axis=0)
        dominant_classes = np.where(class_probs > 0.3)[0]
        if len(dominant_classes) <= 2:
            print(f"  ⚠️ 警告：模型预测过度集中在类别 {dominant_classes}")
        
        # 检查概率分布的方差
        prob_variance = outputs.var(axis=0)
        print(f"  各类别概率方差: {prob_variance}")
        
        if np.any(prob_variance < 0.01):
            low_var_classes = np.where(prob_variance < 0.01)[0]
            print(f"  ⚠️ 警告：类别 {low_var_classes} 的预测方差过低，可能过于固化")

def diagnose_adaptive_centers():
    """诊断自适应中心点"""
    print("\n" + "="*80)
    print("诊断3: 检查自适应中心点")
    print("="*80)
    
    # 创建标签生成器
    label_generator = ImprovedThreeDimensionalLabelGenerator(
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True
    )
    
    # 获取样本并构建分布
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        return
    
    # 按股票分组
    stock_samples_dict = {}
    for sample in all_samples:
        stock_code = sample.get('stock_code', config.STOCK_CODES[0])
        if stock_code not in stock_samples_dict:
            stock_samples_dict[stock_code] = []
        stock_samples_dict[stock_code].append(sample)
    
    # 构建分布
    label_generator.fit_stock_distributions(stock_samples_dict)
    
    print(f"构建了 {len(label_generator.relative_calculator.stock_distributions)} 只股票的分布")
    
    # 检查每只股票的中心点
    for stock_code in config.STOCK_CODES[:3]:  # 只看前3只
        print(f"\n股票 {stock_code} 的自适应中心点:")
        
        for metric_type in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            centers = label_generator.relative_calculator.get_adaptive_centers(stock_code, metric_type)
            print(f"  {metric_type}: {[f'{c:.4f}' for c in centers]}")
        
        # 检查分布统计
        if stock_code in label_generator.relative_calculator.stock_distributions:
            dist = label_generator.relative_calculator.stock_distributions[stock_code]
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                if metric in dist:
                    stats = dist[metric]
                    print(f"  {metric} 统计:")
                    print(f"    均值: {stats['mean']:.4f}")
                    print(f"    标准差: {stats['std']:.4f}")
                    print(f"    分位数: {[f'{q:.4f}' for q in stats['quantiles']]}")

def diagnose_gradient_flow():
    """诊断梯度流问题"""
    print("\n" + "="*80)
    print("诊断4: 检查模型梯度流")
    print("="*80)
    
    model = create_3d_model(config).to(config.DEVICE)
    
    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 检查每层的参数范围
    print("\n各层参数统计:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}:")
            print(f"    形状: {param.shape}")
            print(f"    均值: {param.data.mean():.6f}")
            print(f"    标准差: {param.data.std():.6f}")
            print(f"    最小值: {param.data.min():.6f}")
            print(f"    最大值: {param.data.max():.6f}")
            
            # 检查是否有死神经元
            if 'weight' in name and len(param.shape) >= 2:
                dead_neurons = (param.data.abs() < 1e-6).all(dim=1).sum()
                if dead_neurons > 0:
                    print(f"    ⚠️ 死神经元: {dead_neurons}/{param.shape[0]}")

if __name__ == "__main__":
    print("="*80)
    print("3D模型训练问题诊断工具")
    print("="*80)
    
    # 1. 诊断标签分布
    train_labels, val_labels = diagnose_label_distribution()
    
    # 2. 诊断模型输出
    diagnose_model_outputs()
    
    # 3. 诊断自适应中心点
    diagnose_adaptive_centers()
    
    # 4. 诊断梯度流
    diagnose_gradient_flow()
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)