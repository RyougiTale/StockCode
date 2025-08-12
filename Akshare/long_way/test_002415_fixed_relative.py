#!/usr/bin/env python3
"""
修复版本：正确转换label到相对空间[0,1]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config
    from .data_utils import get_all_samples
    from .model_3d import create_3d_model
    from .dataset_3d import split_samples_by_market_periods
    from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    import torch
except ImportError:
    import config
    from data_utils import get_all_samples
    from model_3d import create_3d_model
    from dataset_3d import split_samples_by_market_periods
    from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    import torch

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from datetime import datetime

def convert_absolute_to_relative(absolute_values, stock_centers):
    """将绝对收益率转换为相对位置[0,1] - 使用训练时一致的逻辑"""
    
    if not stock_centers or 'total_return' not in stock_centers:
        # 回退到简单的分位数映射
        quantiles = np.quantile(absolute_values, [0.1, 0.25, 0.5, 0.75, 0.9])
    else:
        quantiles = stock_centers['total_return']['quantiles']
    
    relative_values = []
    
    for value in absolute_values:
        # 使用与训练时一致的分段线性插值
        if value <= quantiles[0]:
            relative_pos = 0.0
        elif value >= quantiles[-1]:
            relative_pos = 1.0
        else:
            # 分段线性插值
            relative_pos = 0.5  # 默认值
            for i in range(len(quantiles) - 1):
                left, right = quantiles[i], quantiles[i + 1]
                if left <= value <= right and right > left:
                    progress = (value - left) / (right - left)
                    relative_pos = (i + progress) / (len(quantiles) - 1)
                    break
        
        relative_values.append(float(np.clip(relative_pos, 0.0, 1.0)))
    
    return relative_values

def evaluate_samples_fixed(model, samples, name, stock_centers=None):
    """修复版本：正确转换label到相对空间"""
    
    if not samples or len(samples) < 10:
        print(f"❌ {name}: 样本不足")
        return None
    
    predictions = []
    absolute_labels = []
    dates = []
    
    print(f"🔍 评估{name} ({len(samples)}个样本)...")
    
    with torch.no_grad():
        for sample in tqdm(samples, desc=name):
            try:
                # 模型预测
                daily_tensor = torch.FloatTensor(sample['daily']).unsqueeze(0).to(config.DEVICE)
                weekly_tensor = torch.FloatTensor(sample['weekly']).unsqueeze(0).to(config.DEVICE)
                monthly_tensor = torch.FloatTensor(sample['monthly']).unsqueeze(0).to(config.DEVICE)
                
                output = model(daily_tensor, weekly_tensor, monthly_tensor)
                return_probs = torch.exp(output['return']).cpu().numpy()[0]
                
                # 相对空间预测
                relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                predicted_relative = np.sum(return_probs * relative_centers)
                
                # 收集绝对标签，稍后统一转换
                absolute_return = float(sample['label'])
                
                predictions.append(predicted_relative)
                absolute_labels.append(absolute_return)
                dates.append(sample['date'])
                
            except Exception as e:
                continue
    
    if len(predictions) < 10:
        print(f"❌ {name}: 有效预测太少")
        return None
    
    # 将绝对标签转换为相对标签
    relative_labels = convert_absolute_to_relative(absolute_labels, stock_centers)
    
    # 计算指标（现在两者都在[0,1]空间）
    pred_array = np.array(predictions)
    label_array = np.array(relative_labels)
    
    pearson_corr, pearson_p = pearsonr(pred_array, label_array)
    spearman_corr, spearman_p = spearmanr(pred_array, label_array)
    
    # 方向准确率（都在[0,1]空间，以0.5为分界）
    direction_acc = ((pred_array > 0.5) == (label_array > 0.5)).mean()
    
    # 分位数准确率
    pred_quintiles = np.digitize(pred_array, np.quantile(pred_array, [0.2, 0.4, 0.6, 0.8]))
    label_quintiles = np.digitize(label_array, np.quantile(label_array, [0.2, 0.4, 0.6, 0.8]))
    quintile_acc = (pred_quintiles == label_quintiles).mean()
    
    result = {
        'name': name,
        'sample_count': len(predictions),
        'date_range': (min(dates).strftime('%Y-%m-%d'), max(dates).strftime('%Y-%m-%d')),
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'direction_acc': direction_acc,
        'quintile_acc': quintile_acc,
        'pred_stats': {
            'min': pred_array.min(),
            'max': pred_array.max(), 
            'mean': pred_array.mean(),
            'std': pred_array.std()
        },
        'relative_label_stats': {
            'min': label_array.min(),
            'max': label_array.max(),
            'mean': label_array.mean(),
            'std': label_array.std()
        },
        'absolute_label_stats': {
            'min': min(absolute_labels),
            'max': max(absolute_labels),
            'mean': np.mean(absolute_labels),
            'std': np.std(absolute_labels)
        }
    }
    
    return result

def test_002415_fixed():
    """修复版本测试"""
    
    stock_code = '002415'
    model_path = 'long_way/models/enhanced_pretraining/best_loss_top_3.pth'
    
    print("=" * 80)
    print(f"📊 002415 修复版本测试 - 正确的相对空间转换")
    print(f"🤖 模型: {model_path}")
    print(f"🎯 关键修复: 将label正确转换为相对空间[0,1]")
    print("=" * 80)
    
    try:
        # 1. 加载模型
        print("🤖 加载模型...")
        model = create_3d_model(config).to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        
        # 2. 获取全部样本
        print("📊 获取002415全部样本...")
        all_samples, _ = get_all_samples([stock_code])
        
        if not all_samples:
            print("❌ 无法获取样本数据")
            return None
        
        all_samples.sort(key=lambda x: x['date'])
        print(f"✅ 获得 {len(all_samples)} 个样本")
        
        # 3. 建立股票的分布中心点用于相对化
        print("📐 建立分布中心点...")
        label_generator = ImprovedThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        label_generator.fit_stock_distributions({stock_code: all_samples})
        stock_centers = label_generator.relative_calculator.stock_distributions.get(stock_code)
        
        # 4. 按市场时期切分
        print("🔄 按市场时期切分...")
        period_samples_dict = split_samples_by_market_periods(all_samples)
        
        # 5. 评估 - 这次使用修复的函数
        results = []
        
        for period_name, period_samples in period_samples_dict.items():
            if len(period_samples) < 30:
                print(f"⚠️ {period_name}期样本太少，跳过")
                continue
            
            # 时期内部划分
            train_ratio = 0.8
            val_ratio = 0.1
            
            period_size = len(period_samples)
            train_size = int(period_size * train_ratio)
            val_size = int(period_size * val_ratio)
            
            datasets = [
                (period_samples[:train_size], f"{period_name}期-训练集"),
                (period_samples[train_size:train_size + val_size], f"{period_name}期-验证集"),
                (period_samples[train_size + val_size:], f"{period_name}期-测试集")
            ]
            
            for samples, name in datasets:
                if len(samples) >= 10:
                    result = evaluate_samples_fixed(model, samples, name, stock_centers)
                    if result:
                        results.append(result)
        
        # 6. 输出结果
        print("\n" + "=" * 80)
        print("📊 修复后的详细结果")
        print("=" * 80)
        
        for result in results:
            print(f"\n🎯 {result['name']} ({result['sample_count']} 样本):")
            print(f"  📅 时间范围: {result['date_range'][0]} 到 {result['date_range'][1]}")
            print(f"  📈 Pearson相关性: {result['pearson_corr']:7.4f} (p={result['pearson_p']:.4f})")
            print(f"  📊 Spearman相关性: {result['spearman_corr']:7.4f}")
            print(f"  🎯 方向准确率: {result['direction_acc']:7.4f} ({result['direction_acc']*100:.1f}%)")
            print(f"  📋 分位数准确率: {result['quintile_acc']:7.4f} ({result['quintile_acc']*100:.1f}%)")
            print(f"  🔢 预测值(相对): [{result['pred_stats']['min']:.4f}, {result['pred_stats']['max']:.4f}], 均值={result['pred_stats']['mean']:.4f}")
            print(f"  🔢 标签值(相对): [{result['relative_label_stats']['min']:.4f}, {result['relative_label_stats']['max']:.4f}], 均值={result['relative_label_stats']['mean']:.4f}")
            print(f"  🔢 标签值(绝对): [{result['absolute_label_stats']['min']:.4f}, {result['absolute_label_stats']['max']:.4f}], 均值={result['absolute_label_stats']['mean']:.4f}")
        
        # 7. 整体汇总
        if results:
            total_samples = sum(r['sample_count'] for r in results)
            overall_pearson = sum(r['pearson_corr'] * r['sample_count'] for r in results) / total_samples
            overall_direction = sum(r['direction_acc'] * r['sample_count'] for r in results) / total_samples
            
            print(f"\n" + "=" * 80)
            print(f"🏁 修复后的整体结果")
            print(f"=" * 80)
            print(f"📊 整体性能 (基于 {len(results)} 个数据集, 总 {total_samples} 样本):")
            print(f"  总体Pearson相关性: {overall_pearson:.4f}")
            print(f"  总体方向准确率: {overall_direction:.4f} ({overall_direction*100:.1f}%)")
            
            if overall_direction > 0.6:
                print("🎉 修复后方向预测能力优秀!")
            elif overall_direction > 0.55:
                print("✅ 修复后方向预测能力良好")
            elif overall_direction > 0.52:
                print("⚠️ 修复后方向预测能力一般")
            else:
                print("❌ 修复后方向预测能力仍然较差")
            
            print(f"\n💡 关键修复:")
            print(f"  ✅ 将绝对收益率label正确转换为相对位置[0,1]")
            print(f"  ✅ 预测值和真实值现在都在相同的[0,1]空间")
            print(f"  ✅ 消除了空间不匹配导致的误差")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_002415_fixed()
    if results:
        print(f"\n🎊 修复版本测试完成! 共 {len(results)} 个数据集")
    else:
        print("\n💥 修复版本测试失败!")