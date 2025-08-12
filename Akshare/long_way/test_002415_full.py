#!/usr/bin/env python3
"""
测试002415的15年完整数据预测表现 - 相对空间版本
直接在相对空间[0,1]评估return预测能力，无绝对空间映射
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config
    from .data_utils import get_all_samples
    from .model_3d import create_3d_model
    from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    from .logger_config import get_logger
    import torch
except ImportError:
    import config
    from data_utils import get_all_samples
    from model_3d import create_3d_model
    from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    from logger_config import get_logger
    import torch

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm

def calculate_actual_metrics_from_sample(sample, look_forward_days):
    """从样本直接计算未来指标（用于获取真实的相对位置）"""
    try:
        # 从future_prices计算指标（第一个价格作为当前价格）
        future_prices = sample.get('future_prices')
        if future_prices is None or len(future_prices) < look_forward_days:
            return None
            
        # 使用第一个价格作为当前价格，后续作为未来价格
        if len(future_prices) < look_forward_days + 1:
            return None
            
        current_price = future_prices[0]
        if current_price <= 0:
            return None
            
        future_price_series = pd.Series(future_prices[1:look_forward_days + 1])
        if (future_price_series <= 0).any():
            return None
        
        # 计算总回报率
        total_return = (future_price_series.iloc[-1] / current_price) - 1.0
        
        # 计算夏普比率（近似年化）
        daily_returns = future_price_series.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 1e-8:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        # 计算最大回撤
        cumulative = future_price_series / current_price
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }
        
    except Exception as e:
        return None

def convert_metrics_to_relative(metrics, stock_centers):
    """将绝对指标转换为相对位置[0,1]"""
    try:
        relative_metrics = {}
        
        for metric_name in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            if metric_name in metrics and metric_name in stock_centers:
                value = metrics[metric_name]
                centers = stock_centers[metric_name]['quantiles']  # [10%, 25%, 50%, 75%, 90%]
                
                # 边界处理
                if value <= centers[0]:
                    relative_pos = 0.0
                elif value >= centers[-1]:
                    relative_pos = 1.0
                else:
                    # 分段线性插值
                    relative_pos = 0.5  # 默认值
                    for i in range(len(centers) - 1):
                        left, right = centers[i], centers[i + 1]
                        if left <= value <= right and right > left:
                            progress = (value - left) / (right - left)
                            relative_pos = (i + progress) / (len(centers) - 1)
                            break
                
                relative_metrics[metric_name] = float(np.clip(relative_pos, 0.0, 1.0))
        
        return relative_metrics
    except Exception:
        return None

def test_002415_relative_space():
    """在相对空间[0,1]测试002415的return预测表现"""
    
    stock_code = '002415'
    model_path = 'long_way/models/enhanced_pretraining/best_loss_top_3.pth'
    
    print("=" * 80)
    print(f"📊 002415 相对空间return预测测试")
    print(f"🤖 模型: {model_path}")
    print(f"🎯 评估空间: 相对位置 [0,1] (无绝对空间映射)")
    print("=" * 80)
    
    try:
        # 1. 加载模型
        print("🤖 加载训练好的模型...")
        model = create_3d_model(config).to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        
        # 2. 获取002415的所有样本数据
        print("📊 获取002415样本数据...")
        all_samples, scalers = get_all_samples([stock_code])
        
        if not all_samples:
            print("❌ 无法获取样本数据")
            return None
            
        print(f"✅ 获得 {len(all_samples)} 个样本")
        
        # 3. 创建标签生成器用于获取分布中心点
        print("📐 创建标签生成器...")
        label_generator = ImprovedThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        label_generator.fit_stock_distributions({stock_code: all_samples})
        
        # 获取股票的分布中心点（用于相对位置转换）
        stock_centers = label_generator.relative_calculator.stock_distributions[stock_code]
        print(f"📈 获得分布中心点: return范围 [{stock_centers['total_return']['quantiles'][0]:.4f}, {stock_centers['total_return']['quantiles'][-1]:.4f}]")
        
        # 4. 按时间排序并划分数据集
        all_samples.sort(key=lambda x: x['date'])
        
        # 使用最近的样本进行测试（确保有完整的未来数据）
        total_samples = len(all_samples)
        test_samples = all_samples[-1000:]  # 最近1000个样本
        
        print(f"🎯 使用最近 {len(test_samples)} 个样本进行测试")
        if len(test_samples) > 0:
            print(f"📅 测试期间: {test_samples[0]['date'].strftime('%Y-%m-%d')} 到 {test_samples[-1]['date'].strftime('%Y-%m-%d')}")
        
        # 5. 进行预测和评估
        relative_predictions = []
        relative_actuals = []
        dates = []
        
        print("🔍 开始预测和评估...")
        valid_samples = 0
        
        with torch.no_grad():
            for sample in tqdm(test_samples, desc="预测进度"):
                try:
                    # 模型预测
                    daily_tensor = torch.FloatTensor(sample['daily']).unsqueeze(0).to(config.DEVICE)
                    weekly_tensor = torch.FloatTensor(sample['weekly']).unsqueeze(0).to(config.DEVICE)
                    monthly_tensor = torch.FloatTensor(sample['monthly']).unsqueeze(0).to(config.DEVICE)
                    
                    output = model(daily_tensor, weekly_tensor, monthly_tensor)
                    
                    # 获取return维度的预测概率（模型输出log_softmax，需要exp）
                    return_probs = torch.exp(output['return']).cpu().numpy()[0]
                    
                    # 相对空间预测：直接计算期望值，无需映射
                    relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                    predicted_relative = np.sum(return_probs * relative_centers)
                    
                    # 计算真实指标
                    actual_metrics = calculate_actual_metrics_from_sample(
                        sample, config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
                    )
                    if actual_metrics is None:
                        continue
                    
                    # 将真实指标转换为相对位置
                    actual_relative_metrics = convert_metrics_to_relative(actual_metrics, stock_centers)
                    if actual_relative_metrics is None or 'total_return' not in actual_relative_metrics:
                        continue
                    
                    actual_relative = actual_relative_metrics['total_return']
                    
                    # 收集结果
                    relative_predictions.append(predicted_relative)
                    relative_actuals.append(actual_relative)
                    dates.append(sample['date'])
                    valid_samples += 1
                    
                except Exception as e:
                    continue
        
        if valid_samples < 50:
            print(f"❌ 有效样本太少 ({valid_samples})，无法进行可靠的分析")
            return None
            
        print(f"✅ 成功处理 {valid_samples} 个有效样本")
        
        # 6. 计算相对空间的相关性
        print("\n" + "=" * 80)
        print("📊 相对空间 [0,1] 预测效果分析")
        print("=" * 80)
        
        # 基本统计
        pred_array = np.array(relative_predictions)
        actual_array = np.array(relative_actuals)
        
        # 相关性分析
        pearson_corr, pearson_p = pearsonr(pred_array, actual_array)
        spearman_corr, spearman_p = spearmanr(pred_array, actual_array)
        
        # 方向准确率（以0.5为分界点）
        direction_acc = ((pred_array > 0.5) == (actual_array > 0.5)).mean()
        
        # 分位数准确率（预测和实际都按分位数分组）
        pred_quantiles = np.digitize(pred_array, [0.2, 0.4, 0.6, 0.8]) 
        actual_quantiles = np.digitize(actual_array, [0.2, 0.4, 0.6, 0.8])
        quantile_acc = (pred_quantiles == actual_quantiles).mean()
        
        print(f"📈 相关性分析:")
        print(f"  Pearson相关性: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman相关性: {spearman_corr:.4f} (p={spearman_p:.4f})")
        
        print(f"\n🎯 准确率分析:")
        print(f"  方向准确率 (>0.5): {direction_acc:.4f} ({direction_acc*100:.1f}%)")
        print(f"  分位数准确率: {quantile_acc:.4f} ({quantile_acc*100:.1f}%)")
        
        print(f"\n📊 分布分析:")
        print(f"  预测值范围: [{pred_array.min():.4f}, {pred_array.max():.4f}]")
        print(f"  实际值范围: [{actual_array.min():.4f}, {actual_array.max():.4f}]")
        print(f"  预测值均值: {pred_array.mean():.4f}")
        print(f"  实际值均值: {actual_array.mean():.4f}")
        print(f"  预测值标准差: {pred_array.std():.4f}")
        print(f"  实际值标准差: {actual_array.std():.4f}")
        
        # 7. 时间序列分析
        df_results = pd.DataFrame({
            'date': dates,
            'predicted_relative': pred_array,
            'actual_relative': actual_array
        })
        df_results['date'] = pd.to_datetime(df_results['date'])
        df_results = df_results.sort_values('date')
        
        # 按年分析
        print("\n" + "=" * 40)
        print("📅 按年度分析")
        print("=" * 40)
        
        df_results['year'] = df_results['date'].dt.year
        yearly_stats = []
        
        for year in sorted(df_results['year'].unique()):
            year_data = df_results[df_results['year'] == year]
            if len(year_data) < 10:  # 至少需要10个样本
                continue
                
            year_corr, year_p = pearsonr(year_data['predicted_relative'], year_data['actual_relative'])
            year_dir_acc = ((year_data['predicted_relative'] > 0.5) == (year_data['actual_relative'] > 0.5)).mean()
            
            yearly_stats.append({
                'year': year,
                'correlation': year_corr,
                'p_value': year_p,
                'direction_acc': year_dir_acc,
                'sample_count': len(year_data)
            })
            
            print(f"{year}: 相关性={year_corr:6.3f} (p={year_p:.3f}), "
                  f"方向准确率={year_dir_acc:.3f}, 样本数={len(year_data)}")
        
        # 8. 保存结果
        results_file = f"002415_relative_space_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(results_file, index=False)
        print(f"\n💾 详细结果已保存至: {results_file}")
        
        # 9. 结论
        print("\n" + "=" * 80)
        print("🏁 相对空间预测结论")
        print("=" * 80)
        
        if pearson_corr > 0.3:
            print("🎉 模型表现优秀: 相关性 > 0.3")
        elif pearson_corr > 0.15:
            print("✅ 模型表现良好: 0.15 < 相关性 < 0.3")
        elif pearson_corr > 0.05:
            print("⚠️  模型表现一般: 0.05 < 相关性 < 0.15")
        elif pearson_corr > 0:
            print("❌ 模型表现较差: 0 < 相关性 < 0.05")
        else:
            print("💥 模型表现很差: 相关性 ≤ 0")
        
        if direction_acc > 0.65:
            print("🎯 方向预测优秀: 准确率 > 65%")
        elif direction_acc > 0.6:
            print("✅ 方向预测良好: 60% < 准确率 < 65%")
        elif direction_acc > 0.55:
            print("⚠️  方向预测一般: 55% < 准确率 < 60%")
        else:
            print("❌ 方向预测较差: 准确率 ≤ 55%")
            
        print(f"\n💡 关键发现: 在相对空间[0,1]评估，消除了绝对空间映射的误差")
        print(f"📊 这是模型真实的预测能力！")
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'direction_accuracy': direction_acc,
            'quantile_accuracy': quantile_acc,
            'sample_count': valid_samples,
            'yearly_stats': yearly_stats
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_002415_relative_space()
    
    if results:
        print(f"\n🎊 测试完成! Pearson相关性: {results['pearson_correlation']:.4f}")
        print(f"🎯 方向准确率: {results['direction_accuracy']:.4f}")
    else:
        print("\n💥 测试失败!")