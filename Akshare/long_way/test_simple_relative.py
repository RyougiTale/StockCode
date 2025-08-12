#!/usr/bin/env python3
"""
简化版本：直接使用样本的label进行相对空间测试
不重新计算指标，使用训练时一样的标签
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config
    from .data_utils import get_all_samples
    from .model_3d import create_3d_model
    from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    import torch
except ImportError:
    import config
    from data_utils import get_all_samples
    from model_3d import create_3d_model
    from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    import torch

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

def test_simple_relative():
    """简化测试：使用样本自带的label，直接测试相对空间预测"""
    
    stock_code = '002415'
    model_path = 'long_way/models/enhanced_pretraining/best_loss_top_3.pth'
    
    print("=" * 60)
    print(f"简化相对空间测试: {stock_code}")
    print(f"模型: {model_path}")
    print("=" * 60)
    
    # 1. 加载模型
    print("🤖 加载模型...")
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # 2. 获取样本
    all_samples, _ = get_all_samples([stock_code])
    test_samples = sorted(all_samples, key=lambda x: x['date'])[-500:]  # 最近500个样本
    print(f"使用样本数: {len(test_samples)}")
    
    # 3. 预测和收集结果
    relative_predictions = []
    actual_labels = []
    
    print("🔍 开始预测...")
    with torch.no_grad():
        for sample in tqdm(test_samples, desc="预测进度"):
            try:
                # 模型预测
                daily_tensor = torch.FloatTensor(sample['daily']).unsqueeze(0).to(config.DEVICE)
                weekly_tensor = torch.FloatTensor(sample['weekly']).unsqueeze(0).to(config.DEVICE)
                monthly_tensor = torch.FloatTensor(sample['monthly']).unsqueeze(0).to(config.DEVICE)
                
                output = model(daily_tensor, weekly_tensor, monthly_tensor)
                return_probs = torch.exp(output['return']).cpu().numpy()[0]
                
                # 相对空间预测：直接计算期望值
                relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                predicted_relative = np.sum(return_probs * relative_centers)
                
                # 使用样本自带的label作为真实值
                actual_label = float(sample['label'])
                
                relative_predictions.append(predicted_relative)
                actual_labels.append(actual_label)
                
            except Exception as e:
                print(f"样本处理失败: {e}")
                continue
    
    if len(relative_predictions) < 10:
        print("❌ 有效样本太少")
        return
    
    # 4. 分析结果
    pred_array = np.array(relative_predictions)
    actual_array = np.array(actual_labels)
    
    # 相关性分析
    pearson_corr, pearson_p = pearsonr(pred_array, actual_array)
    
    # 方向准确率
    direction_acc = ((pred_array > 0.5) == (actual_array > 0.5)).mean()
    
    print(f"\n结果分析 (基于 {len(relative_predictions)} 个样本):")
    print(f"Pearson相关性: {pearson_corr:.4f} (p={pearson_p:.4f})")
    print(f"方向准确率: {direction_acc:.4f} ({direction_acc*100:.1f}%)")
    
    print(f"\n分布统计:")
    print(f"预测值范围: [{pred_array.min():.4f}, {pred_array.max():.4f}]")
    print(f"实际值范围: [{actual_array.min():.4f}, {actual_array.max():.4f}]")
    print(f"预测值均值: {pred_array.mean():.4f}")
    print(f"实际值均值: {actual_array.mean():.4f}")
    
    # 5. 结论
    print("\n" + "=" * 40)
    print("测试结论")
    print("=" * 40)
    
    if pearson_corr > 0.2:
        print("🎉 相对空间预测效果良好!")
    elif pearson_corr > 0.1:
        print("✅ 相对空间预测有一定效果")
    elif pearson_corr > 0.05:
        print("⚠️  相对空间预测效果一般")
    else:
        print("❌ 相对空间预测效果较差")
    
    if direction_acc > 0.6:
        print("🎯 方向预测能力优秀!")
    elif direction_acc > 0.55:
        print("✅ 方向预测能力良好")
    else:
        print("❌ 方向预测能力不足")
    
    print(f"\n💡 关键发现:")
    print(f"   - 在相对空间[0,1]直接评估，避免了绝对空间映射的误差")
    print(f"   - 这反映了模型在其原生训练空间的真实表现")
    
    return {
        'correlation': pearson_corr,
        'p_value': pearson_p,
        'direction_accuracy': direction_acc,
        'sample_count': len(relative_predictions)
    }

if __name__ == "__main__":
    results = test_simple_relative()
    if results:
        print(f"\n🎊 测试完成! 相关性: {results['correlation']:.4f}, 方向准确率: {results['direction_accuracy']:.4f}")
    else:
        print("\n💥 测试失败!")