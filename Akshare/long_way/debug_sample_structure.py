#!/usr/bin/env python3
"""
调试样本数据结构
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

def debug_sample_structure():
    """调试002415样本的数据结构"""
    
    stock_code = '002415'
    print(f"调试 {stock_code} 样本数据结构...")
    
    # 获取样本数据
    all_samples, scalers = get_all_samples([stock_code])
    
    if not all_samples:
        print("❌ 无法获取样本数据")
        return
    
    print(f"✅ 获得 {len(all_samples)} 个样本")
    
    # 检查第一个样本的结构
    sample = all_samples[0]
    print("\n第一个样本的键:")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {type(value)} shape={value.shape}")
        elif hasattr(value, '__len__') and not isinstance(value, str):
            print(f"  {key}: {type(value)} len={len(value)}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    # 检查最近几个样本，看看数据质量
    recent_samples = sorted(all_samples, key=lambda x: x['date'])[-10:]
    
    print(f"\n最近10个样本的关键字段:")
    for i, sample in enumerate(recent_samples):
        date = sample['date'].strftime('%Y-%m-%d')
        current_close = sample.get('current_close', 'MISSING')
        future_prices = sample.get('future_prices', [])
        future_prices_len = len(future_prices) if future_prices else 0
        
        print(f"  样本{i+1}: {date}, current_close={current_close}, future_prices_len={future_prices_len}")
        
        if future_prices_len > 0 and isinstance(future_prices, list):
            print(f"    future_prices前5个: {future_prices[:5]}")
    
    return all_samples

if __name__ == "__main__":
    samples = debug_sample_structure()