#!/usr/bin/env python3
"""
测试相对空间预测 vs 绝对空间预测的效果差异
验证是否需要绝对位置映射
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

try:
    from . import config
    from .data_utils import get_all_samples
    from .model_3d import create_3d_model
    from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    from .logger_config import get_logger
except ImportError:
    import config
    from data_utils import get_all_samples
    from model_3d import create_3d_model
    from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    from logger_config import get_logger

def reverse_relative_mapping(relative_pos, centers):
    """将相对位置映射回绝对值"""
    centers = np.array(centers)
    scaled_pos = relative_pos * (len(centers) - 1)
    i = int(np.floor(scaled_pos))
    i = min(i, len(centers) - 2)
    progress = scaled_pos - i
    value = centers[i] + progress * (centers[i + 1] - centers[i])
    return value

def test_relative_vs_absolute_prediction():
    """对比相对空间预测 vs 绝对空间预测"""
    
    stock_code = '002415'
    model_path = 'long_way/models/enhanced_pretraining/best_loss_top_3.pth'
    
    print("=" * 80)
    print("🧪 相对空间 vs 绝对空间预测对比")
    print(f"📊 测试股票: {stock_code}")
    print(f"🤖 模型路径: {model_path}")
    print("=" * 80)
    
    try:
        # 1. 加载模型
        print("🤖 加载模型...")
        model = create_3d_model(config).to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        
        # 2. 获取样本数据
        print("📊 获取样本数据...")
        all_samples, _ = get_all_samples([stock_code])
        if not all_samples:
            print("❌ 无法获取样本数据")
            return
        
        print(f"✅ 获得 {len(all_samples)} 个样本")
        
        # 3. 创建标签生成器
        print("🏷️ 创建标签生成器...")
        label_generator = ImprovedThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        label_generator.fit_stock_distributions({stock_code: all_samples})
        
        # 4. 取测试样本（最近500个）
        test_samples = sorted(all_samples, key=lambda x: x['date'])[-500:]
        print(f"🎯 使用最近 {len(test_samples)} 个样本进行测试")
        
        # 5. 进行预测对比
        relative_predictions = []
        absolute_predictions = []  
        relative_actuals = []
        absolute_actuals = []
        
        print("🔍 开始预测...")
        with torch.no_grad():
            for sample in tqdm(test_samples, desc="预测进度"):
                try:
                    # 模型预测
                    daily_tensor = torch.FloatTensor(sample['daily']).unsqueeze(0).to(config.DEVICE)
                    weekly_tensor = torch.FloatTensor(sample['weekly']).unsqueeze(0).to(config.DEVICE)  
                    monthly_tensor = torch.FloatTensor(sample['monthly']).unsqueeze(0).to(config.DEVICE)
                    
                    output = model(daily_tensor, weekly_tensor, monthly_tensor)
                    
                    # 获取return维度的预测概率
                    return_probs = torch.exp(output['return']).cpu().numpy()[0]
                    
                    # 相对空间预测：直接使用相对中心点计算期望值
                    relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                    predicted_relative = np.sum(return_probs * relative_centers)
                    
                    # 绝对空间预测：映射到绝对值
                    stock_centers = label_generator.relative_calculator.stock_distributions[stock_code]['total_return']['quantiles']
                    predicted_absolute = reverse_relative_mapping(predicted_relative, stock_centers)
                    
                    # 获取真实标签
                    labels = label_generator.generate_labels([sample])[0]
                    actual_absolute = labels['metrics']['return']
                    
                    # 计算真实的相对位置（使用与训练一致的convert_to_relative）
                    relative_metrics = label_generator.convert_to_relative(
                        {'total_return': actual_absolute}, stock_code
                    )
                    actual_relative = relative_metrics['total_return'] if relative_metrics else 0.5
                    
                    # 收集结果
                    relative_predictions.append(predicted_relative)
                    absolute_predictions.append(predicted_absolute) 
                    relative_actuals.append(actual_relative)
                    absolute_actuals.append(actual_absolute)
                    
                except Exception as e:
                    print(f"样本处理失败: {e}")
                    continue
        
        if not relative_predictions:
            print("❌ 无有效预测")
            return
            
        print(f"✅ 成功预测 {len(relative_predictions)} 个样本")
        
        # 6. 计算两种方法的相关性
        print("\n" + "=" * 80)
        print("📊 预测效果对比")
        print("=" * 80)
        
        # 相对空间相关性
        relative_corr, relative_p = pearsonr(relative_predictions, relative_actuals)
        relative_dir_acc = ((np.array(relative_predictions) > 0.5) == 
                          (np.array(relative_actuals) > 0.5)).mean()
        
        # 绝对空间相关性  
        absolute_corr, absolute_p = pearsonr(absolute_predictions, absolute_actuals)
        absolute_dir_acc = ((np.array(absolute_predictions) > 0) == 
                          (np.array(absolute_actuals) > 0)).mean()
        
        print("🎯 相对空间预测 (模型原生空间):")
        print(f"   相关性: {relative_corr:.4f} (p={relative_p:.4f})")
        print(f"   方向准确率: {relative_dir_acc:.4f} ({relative_dir_acc*100:.1f}%)")
        print(f"   预测值范围: [{min(relative_predictions):.4f}, {max(relative_predictions):.4f}]")
        print(f"   实际值范围: [{min(relative_actuals):.4f}, {max(relative_actuals):.4f}]")
        
        print("\n🎯 绝对空间预测 (映射后的空间):")
        print(f"   相关性: {absolute_corr:.4f} (p={absolute_p:.4f})")  
        print(f"   方向准确率: {absolute_dir_acc:.4f} ({absolute_dir_acc*100:.1f}%)")
        print(f"   预测值范围: [{min(absolute_predictions):.4f}, {max(absolute_predictions):.4f}]")
        print(f"   实际值范围: [{min(absolute_actuals):.4f}, {max(absolute_actuals):.4f}]")
        
        # 7. 结论
        print("\n" + "=" * 80)
        print("🏁 结论")  
        print("=" * 80)
        
        if relative_corr > absolute_corr:
            improvement = relative_corr - absolute_corr
            print(f"✅ 相对空间预测更好！相关性提升: {improvement:.4f}")
            print("💡 建议：直接在相对空间评估模型效果，无需映射到绝对值")
        else:
            decline = absolute_corr - relative_corr  
            print(f"⚠️ 绝对空间预测更好，相关性优势: {decline:.4f}")
            print("💡 绝对值映射是有必要的")
            
        if abs(relative_corr - absolute_corr) < 0.01:
            print("🔄 两种方法效果相近，映射影响较小")
            
        return {
            'relative': {'correlation': relative_corr, 'p_value': relative_p, 'direction_acc': relative_dir_acc},
            'absolute': {'correlation': absolute_corr, 'p_value': absolute_p, 'direction_acc': absolute_dir_acc},
            'sample_count': len(relative_predictions)
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_relative_vs_absolute_prediction()
    
    if results:
        print(f"\n🎊 测试完成!")
        print(f"相对空间相关性: {results['relative']['correlation']:.4f}")
        print(f"绝对空间相关性: {results['absolute']['correlation']:.4f}")
    else:
        print("\n💥 测试失败!")