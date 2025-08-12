#!/usr/bin/env python3
"""
调试样本中的label到底是什么含义
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config
    from .data_utils import get_all_samples
except ImportError:
    import config
    from data_utils import get_all_samples

import numpy as np

def debug_label_meaning():
    """调试label的含义"""
    
    stock_code = '002415'
    print(f"调试 {stock_code} 的label含义...")
    
    # 获取样本数据
    all_samples, scalers = get_all_samples([stock_code])
    
    if not all_samples:
        print("❌ 无法获取样本数据")
        return
    
    print(f"✅ 获得 {len(all_samples)} 个样本")
    
    # 分析label的分布
    labels = [float(sample['label']) for sample in all_samples[:100]]  # 取前100个样本
    
    print(f"\nLabel统计 (前100个样本):")
    print(f"  范围: [{min(labels):.4f}, {max(labels):.4f}]")
    print(f"  均值: {np.mean(labels):.4f}")
    print(f"  标准差: {np.std(labels):.4f}")
    print(f"  中位数: {np.median(labels):.4f}")
    
    # 检查是否在[0,1]范围内
    in_range_01 = [l for l in labels if 0 <= l <= 1]
    print(f"  在[0,1]范围内的数量: {len(in_range_01)}/{len(labels)} ({len(in_range_01)/len(labels)*100:.1f}%)")
    
    # 检查几个具体样本的其他信息
    print(f"\n前5个样本详情:")
    for i, sample in enumerate(all_samples[:5]):
        print(f"  样本{i+1}:")
        print(f"    日期: {sample['date']}")
        print(f"    label: {sample['label']:.4f}")
        print(f"    future_prices长度: {len(sample['future_prices'])}")
        if len(sample['future_prices']) >= 2:
            print(f"    future_prices前3个: {sample['future_prices'][:3]}")
    
    # 如果label看起来像收益率，我们验证一下
    print(f"\n分析label是否为收益率:")
    if abs(np.mean(labels)) < 0.5 and min(labels) > -1:
        print(f"  ✅ label看起来像收益率 (均值接近0，最小值>-1)")
    else:
        print(f"  ❌ label不太像收益率")
    
    # 检查label是否需要转换为[0,1]空间
    print(f"\n建议:")
    if max(labels) <= 1 and min(labels) >= 0:
        print(f"  ✅ label已经在[0,1]空间，可以直接使用")
    else:
        print(f"  ❌ label不在[0,1]空间，需要转换为相对位置")
        print(f"  💡 应该使用训练时的相对化逻辑转换label")

if __name__ == "__main__":
    debug_label_meaning()