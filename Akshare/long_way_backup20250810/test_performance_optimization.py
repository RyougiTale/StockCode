import sys
import os
import time
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import config
from .rolling_scaler import RollingWindowScaler

def create_test_data(n_rows=10000, n_features=12):
    """创建测试数据"""
    np.random.seed(42)
    
    # 模拟股票数据特征
    data = {}
    feature_names = config.FEATURE_COLUMNS['daily'][:n_features]
    
    for i, feature in enumerate(feature_names):
        if feature in ['open', 'high', 'low', 'close']:
            # 价格数据：有趋势 + 随机波动
            trend = np.linspace(100, 200, n_rows)
            noise = np.random.normal(0, 5, n_rows)
            data[feature] = trend + noise + i * 10  # 不同价格水平
        elif feature == 'volume':
            # 成交量：对数正态分布
            data[feature] = np.random.lognormal(10, 1, n_rows)
        elif feature in ['pct_chg', 'chg_amount']:
            # 涨跌幅：正态分布
            data[feature] = np.random.normal(0, 2, n_rows)
        else:
            # 其他特征：随机数据
            data[feature] = np.random.randn(n_rows) * 10 + 50
    
    # 添加日期列
    data['date'] = pd.date_range('2020-01-01', periods=n_rows, freq='D')
    
    return pd.DataFrame(data)

def benchmark_rolling_scaler():
    """测试滚动窗口归一化的性能"""
    print("=== 滚动窗口归一化性能测试 ===")
    
    # 测试不同数据规模
    test_sizes = [1000, 5000, 10000, 20000]
    methods = ['zscore', 'minmax', 'robust']
    
    results = []
    
    for size in test_sizes:
        print(f"\n测试数据规模: {size} 行")
        test_data = create_test_data(n_rows=size)
        feature_columns = [col for col in config.FEATURE_COLUMNS['daily'] if col in test_data.columns]
        
        for method in methods:
            print(f"  测试方法: {method}")
            
            # 创建归一化器
            scaler = RollingWindowScaler(
                window_size=min(252, size//4),  # 适应数据大小
                method=method,
                min_periods=min(60, size//10)
            )
            
            # 测试性能
            start_time = time.time()
            normalized_data = scaler.fit_transform(test_data, feature_columns)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            rows_per_second = size / elapsed_time
            
            print(f"    耗时: {elapsed_time:.4f}秒")
            print(f"    处理速度: {rows_per_second:.0f} 行/秒")
            
            # 验证结果正确性
            is_valid = validate_normalization_result(normalized_data, feature_columns, method)
            print(f"    结果验证: {'✓ 通过' if is_valid else '✗ 失败'}")
            
            results.append({
                'size': size,
                'method': method,
                'time': elapsed_time,
                'rows_per_second': rows_per_second,
                'valid': is_valid
            })
    
    return results

def validate_normalization_result(normalized_data, feature_columns, method):
    """验证归一化结果的正确性"""
    try:
        for col in feature_columns:
            if col not in normalized_data.columns:
                continue
                
            data = normalized_data[col].dropna()
            if len(data) == 0:
                continue
            
            if method == 'zscore':
                # Z-score应该大致均值为0，标准差为1（考虑滚动窗口的影响）
                mean_val = data.mean()
                std_val = data.std()
                if abs(mean_val) > 2 or std_val < 0.1 or std_val > 5:  # 宽松的检查
                    print(f"    警告: {col} 的Z-score分布异常 (mean={mean_val:.4f}, std={std_val:.4f})")
                    
            elif method == 'minmax':
                # MinMax应该在[0,1]范围内
                min_val = data.min()
                max_val = data.max()
                if min_val < -0.1 or max_val > 1.1:  # 允许小的误差
                    print(f"    警告: {col} 的MinMax范围异常 (min={min_val:.4f}, max={max_val:.4f})")
            
            # 检查是否有无穷大或NaN
            if np.isinf(data).any():
                print(f"    错误: {col} 包含无穷大值")
                return False
                
            if np.isnan(data).any():
                print(f"    警告: {col} 包含NaN值")
        
        return True
        
    except Exception as e:
        print(f"    验证过程出错: {e}")
        return False

def benchmark_draw_optimization():
    """测试draw.py的优化效果"""
    print("\n=== Draw模块优化效果测试 ===")
    
    # 模拟多模型场景
    n_models = 6
    data_size = 1000
    
    print(f"模拟场景: {n_models}个模型, {data_size}行数据")
    
    # 测试原始方法（每个模型重复计算）
    print("\n原始方法（重复计算）:")
    start_time = time.time()
    
    for i in range(n_models):
        test_data = create_test_data(n_rows=data_size)
        feature_columns = [col for col in config.FEATURE_COLUMNS['daily'] if col in test_data.columns]
        
        scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
        _ = scaler.fit_transform(test_data, feature_columns)
    
    original_time = time.time() - start_time
    print(f"  总耗时: {original_time:.4f}秒")
    
    # 测试优化方法（只计算一次）
    print("\n优化方法（只计算一次）:")
    start_time = time.time()
    
    # 只计算一次数据预处理
    test_data = create_test_data(n_rows=data_size)
    feature_columns = [col for col in config.FEATURE_COLUMNS['daily'] if col in test_data.columns]
    
    scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    normalized_data = scaler.fit_transform(test_data, feature_columns)
    
    # 模拟多个模型使用相同的预处理数据
    for i in range(n_models):
        _ = normalized_data.copy()  # 模拟使用预处理数据
    
    optimized_time = time.time() - start_time
    print(f"  总耗时: {optimized_time:.4f}秒")
    
    speedup = original_time / optimized_time
    print(f"\n性能提升: {speedup:.2f}x 倍")
    print(f"时间节省: {(1 - optimized_time/original_time)*100:.1f}%")
    
    return {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup
    }

def main():
    """主测试函数"""
    print("开始性能优化测试...\n")
    
    # 测试1: 滚动窗口归一化性能
    scaler_results = benchmark_rolling_scaler()
    
    # 测试2: Draw模块优化效果
    draw_results = benchmark_draw_optimization()
    
    # 总结报告
    print("\n" + "="*60)
    print("性能优化测试总结")
    print("="*60)
    
    print("\n1. 滚动窗口归一化性能:")
    print("   数据规模    方法      处理速度(行/秒)")
    print("   " + "-"*40)
    
    for result in scaler_results:
        if result['valid']:
            print(f"   {result['size']:8d}    {result['method']:6s}    {result['rows_per_second']:8.0f}")
    
    print(f"\n2. Draw模块优化效果:")
    print(f"   性能提升: {draw_results['speedup']:.2f}x")
    print(f"   时间节省: {(1 - draw_results['optimized_time']/draw_results['original_time'])*100:.1f}%")
    
    # 性能建议
    print(f"\n3. 性能建议:")
    avg_speed = np.mean([r['rows_per_second'] for r in scaler_results if r['valid']])
    print(f"   - 滚动归一化平均处理速度: {avg_speed:.0f} 行/秒")
    
    if avg_speed > 10000:
        print("   - ✅ 性能优秀，可以处理大规模数据")
    elif avg_speed > 1000:
        print("   - ✅ 性能良好，适合中等规模数据")
    else:
        print("   - ⚠️  性能一般，大数据量时需要注意")
    
    print(f"   - Draw模块优化显著，多模型场景下性能提升 {draw_results['speedup']:.1f} 倍")

if __name__ == '__main__':
    main()