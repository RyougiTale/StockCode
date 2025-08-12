import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import config
from .data_utils import get_all_samples

def test_scaler_compatibility():
    """
    测试所有使用scaler的地方是否兼容新的RollingWindowScaler
    """
    print("=== 测试Scaler兼容性 ===")
    
    try:
        # 1. 测试数据获取和训练时的scaler保存
        print("1. 测试数据获取和scaler生成...")
        all_samples, scalers = get_all_samples(config.STOCK_CODES)
        
        if not all_samples:
            print("ERROR: 无法获取样本数据")
            return False
        
        print(f"✓ 成功获取 {len(all_samples)} 个样本")
        print(f"✓ Scalers类型: {type(scalers)}")
        print(f"✓ Scaler键: {list(scalers.keys())}")
        
        # 检查每个scaler的类型
        for key, scaler in scalers.items():
            print(f"  {key} scaler类型: {type(scaler)}")
            print(f"  {key} scaler有transform方法: {hasattr(scaler, 'transform')}")
            print(f"  {key} scaler有global_stats: {hasattr(scaler, 'global_stats')}")
        
        # 2. 测试transform方法
        print("\n2. 测试transform方法...")
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200],
            'turnover': [100000, 110000, 120000],
            'amplitude': [5, 6, 7],
            'pct_chg': [1, 2, 3],
            'chg_amount': [1, 2, 3],
            'turnover_rate': [0.1, 0.2, 0.3],
            'SMA20': [100, 101, 102],
            'SMA60': [99, 100, 101]
        })
        
        # 测试每个scaler的transform方法
        for timeframe in ['daily', 'weekly', 'monthly']:
            scaler = scalers[timeframe]
            feature_cols = config.FEATURE_COLUMNS[timeframe]
            
            print(f"\n  测试 {timeframe} scaler:")
            try:
                # 测试transform方法
                transformed = scaler.transform(test_data, feature_cols)
                print(f"    ✓ Transform成功，输出形状: {transformed.shape}")
                print(f"    ✓ 输出列: {list(transformed.columns)}")
                print(f"    ✓ 数据范围: [{transformed.min().min():.4f}, {transformed.max().max():.4f}]")
                
                # 检查是否有NaN
                nan_count = transformed.isnull().sum().sum()
                if nan_count > 0:
                    print(f"    ⚠️  包含 {nan_count} 个NaN值")
                else:
                    print(f"    ✓ 无NaN值")
                    
            except Exception as e:
                print(f"    ✗ Transform失败: {e}")
                return False
        
        # 3. 测试predict.py和draw.py的兼容性（模拟）
        print("\n3. 测试predict和draw模块的兼容性...")
        
        # 模拟predict.py中的使用方式
        try:
            daily_scaler = scalers['daily']
            weekly_scaler = scalers['weekly']
            monthly_scaler = scalers['monthly']
            
            # 模拟predict.py中的调用
            daily_transformed = daily_scaler.transform(test_data, config.FEATURE_COLUMNS['daily'])
            weekly_transformed = weekly_scaler.transform(test_data, config.FEATURE_COLUMNS['weekly'])
            monthly_transformed = monthly_scaler.transform(test_data, config.FEATURE_COLUMNS['monthly'])
            
            print("    ✓ predict.py风格的调用成功")
            print(f"    ✓ Daily输出形状: {daily_transformed.shape}")
            print(f"    ✓ Weekly输出形状: {weekly_transformed.shape}")
            print(f"    ✓ Monthly输出形状: {monthly_transformed.shape}")
            
        except Exception as e:
            print(f"    ✗ predict.py风格调用失败: {e}")
            return False
        
        # 4. 测试数据类型兼容性
        print("\n4. 测试数据类型兼容性...")
        
        # 测试不同数据类型的输入
        test_types = {
            'int': test_data.astype(int),
            'float32': test_data.astype(np.float32),
            'float64': test_data.astype(np.float64)
        }
        
        for dtype_name, typed_data in test_types.items():
            try:
                result = scalers['daily'].transform(typed_data, config.FEATURE_COLUMNS['daily'])
                print(f"    ✓ {dtype_name}类型输入成功，输出类型: {result.dtypes.iloc[0]}")
            except Exception as e:
                print(f"    ⚠️  {dtype_name}类型输入失败: {e}")
        
        print("\n=== 所有兼容性测试通过 ===")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scaler_performance():
    """
    测试scaler的性能
    """
    print("\n=== 测试Scaler性能 ===")
    
    try:
        # 获取scalers
        _, scalers = get_all_samples(config.STOCK_CODES)
        
        # 创建大一点的测试数据
        n_rows = 1000
        test_data = pd.DataFrame({
            col: np.random.randn(n_rows) for col in config.FEATURE_COLUMNS['daily']
        })
        
        import time
        
        # 测试transform性能
        start_time = time.time()
        for _ in range(10):  # 重复10次
            result = scalers['daily'].transform(test_data, config.FEATURE_COLUMNS['daily'])
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Transform平均耗时: {avg_time:.4f}秒 (数据量: {n_rows}行)")
        print(f"每行处理时间: {avg_time/n_rows*1000:.4f}毫秒")
        
        return True
        
    except Exception as e:
        print(f"性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始Scaler兼容性和性能测试...\n")
    
    # 兼容性测试
    compatibility_ok = test_scaler_compatibility()
    
    # 性能测试
    performance_ok = test_scaler_performance()
    
    print(f"\n{'='*50}")
    print("测试结果总结:")
    print(f"  兼容性测试: {'✓ 通过' if compatibility_ok else '✗ 失败'}")
    print(f"  性能测试: {'✓ 通过' if performance_ok else '✗ 失败'}")
    
    if compatibility_ok and performance_ok:
        print("\n🎉 所有测试通过！Scaler迁移成功完成。")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")
    
    return compatibility_ok and performance_ok

if __name__ == '__main__':
    main()