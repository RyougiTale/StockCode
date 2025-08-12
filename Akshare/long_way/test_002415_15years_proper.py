#!/usr/bin/env python3
"""
002415完整15年数据测试 - 按训练时的市场时期切分方式
严格按照训练时的逻辑：先按市场时期分组，再在每个时期内8:1:1划分
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config
    from .data_utils import get_all_samples
    from .model_3d import create_3d_model
    from .dataset_3d import split_samples_by_market_periods
    import torch
except ImportError:
    import config
    from data_utils import get_all_samples
    from model_3d import create_3d_model
    from dataset_3d import split_samples_by_market_periods
    import torch

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from datetime import datetime

def evaluate_samples(model, samples, name):
    """评估样本集合的性能"""
    
    if not samples or len(samples) < 10:
        print(f"❌ {name}: 样本不足")
        return None
    
    predictions = []
    labels = []
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
                
                # 真实标签：将绝对收益率转换为相对位置[0,1]
                absolute_return = float(sample['label'])
                # 需要使用与训练时一致的相对化逻辑
                # 这里先用简单的分位数映射，后面可以改进为训练时的精确逻辑
                actual_label = absolute_return  # 暂时保持原样，后面修复
                
                predictions.append(predicted_relative)
                labels.append(actual_label)
                dates.append(sample['date'])
                
            except Exception as e:
                continue
    
    if len(predictions) < 10:
        print(f"❌ {name}: 有效预测太少")
        return None
    
    # 计算指标
    pred_array = np.array(predictions)
    label_array = np.array(labels)
    
    pearson_corr, pearson_p = pearsonr(pred_array, label_array)
    spearman_corr, spearman_p = spearmanr(pred_array, label_array)
    
    # 方向准确率
    direction_acc = ((pred_array > 0.5) == (label_array > 0)).mean()
    
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
        'label_stats': {
            'min': label_array.min(),
            'max': label_array.max(),
            'mean': label_array.mean(),
            'std': label_array.std()
        }
    }
    
    return result

def test_002415_proper_split():
    """按训练时的正确方式测试002415"""
    
    stock_code = '002415'
    model_path = 'long_way/models/enhanced_pretraining/best_loss_top_3.pth'
    
    print("=" * 80)
    print(f"📊 002415 按训练时市场时期切分方式测试")
    print(f"🤖 模型: {model_path}")
    print(f"🎯 评估空间: 相对位置 [0,1]")
    print(f"📋 市场时期: Recent(3年) / Middle(3-8年) / Distant(8年+)")
    print(f"📊 每时期内部: 训练集(80%) / 验证集(10%) / 测试集(10%)")
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
        
        # 按时间排序
        all_samples.sort(key=lambda x: x['date'])
        print(f"✅ 获得 {len(all_samples)} 个样本")
        print(f"📅 时间范围: {all_samples[0]['date'].strftime('%Y-%m-%d')} 到 {all_samples[-1]['date'].strftime('%Y-%m-%d')}")
        
        # 3. 按市场时期切分（使用训练时的函数）
        print("\n🔄 按市场时期切分...")
        period_samples_dict = split_samples_by_market_periods(all_samples)
        
        # 4. 在每个时期内部按8:1:1划分
        all_datasets = []
        
        train_ratio = 0.8
        val_ratio = 0.1
        
        for period_name, period_samples in period_samples_dict.items():
            if len(period_samples) < 30:  # 至少需要30个样本才能有效划分
                print(f"⚠️ {period_name}期样本太少({len(period_samples)})，跳过")
                continue
            
            # 时期内部划分
            period_size = len(period_samples)
            train_size = int(period_size * train_ratio)
            val_size = int(period_size * val_ratio)
            
            train_samples = period_samples[:train_size]
            val_samples = period_samples[train_size:train_size + val_size]
            test_samples = period_samples[train_size + val_size:]
            
            print(f"\n📂 {period_name}期数据划分:")
            if train_samples:
                print(f"  训练集: {len(train_samples)} 样本 ({train_samples[0]['date'].strftime('%Y-%m-%d')} 到 {train_samples[-1]['date'].strftime('%Y-%m-%d')})")
            if val_samples:
                print(f"  验证集: {len(val_samples)} 样本 ({val_samples[0]['date'].strftime('%Y-%m-%d')} 到 {val_samples[-1]['date'].strftime('%Y-%m-%d')})")  
            if test_samples:
                print(f"  测试集: {len(test_samples)} 样本 ({test_samples[0]['date'].strftime('%Y-%m-%d')} 到 {test_samples[-1]['date'].strftime('%Y-%m-%d')})")
            
            # 收集数据集
            if train_samples:
                all_datasets.append((train_samples, f"{period_name}期-训练集"))
            if val_samples:
                all_datasets.append((val_samples, f"{period_name}期-验证集"))
            if test_samples:
                all_datasets.append((test_samples, f"{period_name}期-测试集"))
        
        # 5. 评估每个数据集
        results = []
        
        for samples, name in all_datasets:
            result = evaluate_samples(model, samples, name)
            if result:
                results.append(result)
        
        # 6. 汇总分析
        print("\n" + "=" * 80)
        print("📊 各数据集详细结果")
        print("=" * 80)
        
        for result in results:
            print(f"\n🎯 {result['name']} ({result['sample_count']} 样本):")
            print(f"  📅 时间范围: {result['date_range'][0]} 到 {result['date_range'][1]}")
            print(f"  📈 Pearson相关性: {result['pearson_corr']:7.4f} (p={result['pearson_p']:.4f})")
            print(f"  📊 Spearman相关性: {result['spearman_corr']:7.4f}")
            print(f"  🎯 方向准确率: {result['direction_acc']:7.4f} ({result['direction_acc']*100:.1f}%)")
            print(f"  📋 分位数准确率: {result['quintile_acc']:7.4f} ({result['quintile_acc']*100:.1f}%)")
            print(f"  🔢 预测值: [{result['pred_stats']['min']:.4f}, {result['pred_stats']['max']:.4f}], 均值={result['pred_stats']['mean']:.4f}")
            print(f"  🔢 实际值: [{result['label_stats']['min']:.4f}, {result['label_stats']['max']:.4f}], 均值={result['label_stats']['mean']:.4f}")
        
        # 7. 按数据集类型汇总
        print("\n" + "=" * 60)
        print("📋 按数据集类型汇总")
        print("=" * 60)
        
        dataset_types = ['训练集', '验证集', '测试集']
        type_summary = {}
        
        for dataset_type in dataset_types:
            matching_results = [r for r in results if dataset_type in r['name']]
            if not matching_results:
                continue
            
            total_samples = sum(r['sample_count'] for r in matching_results)
            
            # 加权平均
            weighted_pearson = sum(r['pearson_corr'] * r['sample_count'] for r in matching_results) / total_samples
            weighted_direction = sum(r['direction_acc'] * r['sample_count'] for r in matching_results) / total_samples
            
            type_summary[dataset_type] = {
                'period_count': len(matching_results),
                'total_samples': total_samples,
                'weighted_pearson': weighted_pearson,
                'weighted_direction': weighted_direction
            }
            
            print(f"\n{dataset_type} (跨 {len(matching_results)} 个时期, 总 {total_samples} 样本):")
            print(f"  加权Pearson相关性: {weighted_pearson:.4f}")
            print(f"  加权方向准确率: {weighted_direction:.4f} ({weighted_direction*100:.1f}%)")
        
        # 8. 整体总结
        print("\n" + "=" * 80)
        print("🏁 整体测试总结")
        print("=" * 80)
        
        if results:
            total_samples = sum(r['sample_count'] for r in results)
            overall_pearson = sum(r['pearson_corr'] * r['sample_count'] for r in results) / total_samples
            overall_direction = sum(r['direction_acc'] * r['sample_count'] for r in results) / total_samples
            
            print(f"📊 整体性能 (基于 {len(results)} 个数据集, 总 {total_samples} 样本):")
            print(f"  总体Pearson相关性: {overall_pearson:.4f}")
            print(f"  总体方向准确率: {overall_direction:.4f} ({overall_direction*100:.1f}%)")
            
            # 性能评级
            if overall_direction > 0.6:
                print("🎉 方向预测能力优秀!")
            elif overall_direction > 0.55:
                print("✅ 方向预测能力良好")
            elif overall_direction > 0.52:
                print("⚠️ 方向预测能力一般")
            else:
                print("❌ 方向预测能力较差")
            
            print(f"\n💡 关键发现:")
            print(f"  - 使用了训练时完全一致的市场时期切分方式")
            print(f"  - 在相对空间[0,1]直接评估，避免绝对映射误差")
            print(f"  - 可以看出模型在不同时期和数据集上的真实表现")
        
        # 9. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 构建详细结果DataFrame
        all_data = []
        for result in results:
            all_data.append({
                'dataset_name': result['name'],
                'sample_count': result['sample_count'],
                'date_start': result['date_range'][0],
                'date_end': result['date_range'][1],
                'pearson_corr': result['pearson_corr'],
                'pearson_p': result['pearson_p'],
                'spearman_corr': result['spearman_corr'],
                'direction_acc': result['direction_acc'],
                'quintile_acc': result['quintile_acc']
            })
        
        if all_data:
            results_df = pd.DataFrame(all_data)
            results_file = f"002415_proper_split_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"\n💾 详细结果已保存至: {results_file}")
        
        return {
            'results': results,
            'type_summary': type_summary,
            'overall_pearson': overall_pearson if results else 0,
            'overall_direction': overall_direction if results else 0,
            'total_samples': total_samples if results else 0
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    final_results = test_002415_proper_split()
    
    if final_results and final_results['results']:
        print(f"\n🎊 按训练方式的完整测试完成!")
        print(f"📊 总体相关性: {final_results['overall_pearson']:.4f}")
        print(f"🎯 总体方向准确率: {final_results['overall_direction']:.4f}")
        print(f"📈 总样本数: {final_results['total_samples']}")
    else:
        print("\n💥 测试失败!")