import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import config
from .data_utils import get_all_samples
from .rolling_scaler import RollingWindowScaler

def test_rolling_vs_global_normalization():
    """
    测试滚动窗口归一化 vs 全局归一化的效果
    """
    print("=== 测试滚动窗口归一化效果 ===")
    
    # 获取数据
    print("1. 获取原始数据...")
    all_samples, scalers = get_all_samples(config.STOCK_CODES)
    
    if not all_samples:
        print("ERROR: 无法获取样本数据")
        return
    
    print(f"总样本数: {len(all_samples)}")
    
    # 提取时间序列数据进行对比
    dates = [s['date'] for s in all_samples]
    daily_data = np.array([s['daily'] for s in all_samples])
    
    print(f"数据形状: {daily_data.shape}")
    print(f"时间范围: {min(dates)} 到 {max(dates)}")
    
    # 分析归一化效果
    analyze_normalization_quality(daily_data, dates)
    
    return all_samples

def analyze_normalization_quality(data, dates):
    """
    分析归一化质量
    """
    print("\n2. 分析归一化质量...")
    
    # 选择几个特征进行分析
    feature_names = config.FEATURE_COLUMNS['daily']
    n_features = min(4, len(feature_names))  # 最多分析4个特征
    
    for i in range(n_features):
        feature_name = feature_names[i]
        feature_data = data[:, :, i].flatten()  # 展平所有时间步的数据
        
        print(f"\n特征: {feature_name}")
        print(f"  均值: {feature_data.mean():.4f}")
        print(f"  标准差: {feature_data.std():.4f}")
        print(f"  最小值: {feature_data.min():.4f}")
        print(f"  最大值: {feature_data.max():.4f}")
        
        # 检查分布的稳定性
        analyze_temporal_stability(data[:, :, i], dates, feature_name)

def analyze_temporal_stability(feature_data, dates, feature_name):
    """
    分析时间稳定性
    """
    # 按时间段分析统计特性
    n_samples = len(feature_data)
    segment_size = n_samples // 4  # 分成4个时间段
    
    print(f"  时间稳定性分析 ({feature_name}):")
    
    for seg in range(4):
        start_idx = seg * segment_size
        end_idx = (seg + 1) * segment_size if seg < 3 else n_samples
        
        segment_data = feature_data[start_idx:end_idx].flatten()
        segment_dates = dates[start_idx:end_idx]
        
        print(f"    时间段 {seg+1} ({segment_dates[0].strftime('%Y-%m')} 到 {segment_dates[-1].strftime('%Y-%m')}):")
        print(f"      均值: {segment_data.mean():.4f}, 标准差: {segment_data.std():.4f}")

def compare_with_different_window_sizes():
    """
    比较不同窗口大小的效果
    """
    print("\n=== 比较不同窗口大小的效果 ===")
    
    # 创建测试数据
    np.random.seed(42)
    n_points = 1000
    
    # 模拟市场数据：前半段低波动，后半段高波动
    low_vol_data = np.random.normal(100, 5, n_points//2)
    high_vol_data = np.random.normal(120, 20, n_points//2)
    test_data = np.concatenate([low_vol_data, high_vol_data])
    
    # 添加趋势
    trend = np.linspace(0, 50, n_points)
    test_data += trend
    
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_points, freq='D'),
        'price': test_data
    })
    
    # 测试不同窗口大小
    window_sizes = [60, 120, 252, 500]
    
    results = {}
    for window_size in window_sizes:
        scaler = RollingWindowScaler(window_size=window_size, method='zscore')
        normalized_df = scaler.fit_transform(df, ['price'])
        results[window_size] = normalized_df['price'].values
        
        print(f"窗口大小 {window_size}:")
        print(f"  前半段标准差: {normalized_df['price'][:n_points//2].std():.4f}")
        print(f"  后半段标准差: {normalized_df['price'][n_points//2:].std():.4f}")
    
    return results

def test_multi_stock_scenario():
    """
    测试多股票场景
    """
    print("\n=== 测试多股票场景 ===")
    
    # 模拟多只股票数据
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # 创建不同特性的股票
    stocks = {
        'LOW_VOL': {  # 低波动股票（如银行股）
            'price': 100 + np.cumsum(np.random.normal(0, 0.5, n_days)),
            'volume': np.random.lognormal(8, 0.5, n_days)
        },
        'HIGH_VOL': {  # 高波动股票（如科技股）
            'price': 50 + np.cumsum(np.random.normal(0, 2, n_days)),
            'volume': np.random.lognormal(10, 1, n_days)
        },
        'CYCLICAL': {  # 周期性股票
            'price': 80 + 20 * np.sin(np.arange(n_days) * 2 * np.pi / 252) + np.cumsum(np.random.normal(0, 1, n_days)),
            'volume': np.random.lognormal(9, 0.8, n_days)
        }
    }
    
    # 为每只股票创建DataFrame
    stock_dfs = {}
    for stock_name, data in stocks.items():
        stock_dfs[stock_name] = pd.DataFrame({
            'date': dates,
            'price': data['price'],
            'volume': data['volume'],
            'pct_chg': np.concatenate([[0], np.diff(data['price']) / data['price'][:-1] * 100])
        })
    
    # 使用多股票滚动归一化
    from .rolling_scaler import MultiStockRollingScaler
    
    multi_scaler = MultiStockRollingScaler(window_size=60, method='zscore')
    normalized_stocks = multi_scaler.fit_transform_multi_stock(
        stock_dfs, ['price', 'volume', 'pct_chg']
    )
    
    # 分析结果
    print("多股票归一化结果:")
    for stock_name in stocks.keys():
        original = stock_dfs[stock_name]
        normalized = normalized_stocks[stock_name]
        
        print(f"\n{stock_name}:")
        print(f"  原始价格标准差: {original['price'].std():.2f}")
        print(f"  归一化后价格标准差: {normalized['price'].std():.4f}")
        print(f"  原始成交量标准差: {original['volume'].std():.0f}")
        print(f"  归一化后成交量标准差: {normalized['volume'].std():.4f}")
    
    return normalized_stocks

def main():
    """主测试函数"""
    try:
        # 测试1: 实际数据的滚动归一化
        test_rolling_vs_global_normalization()
        
        # 测试2: 不同窗口大小比较
        compare_with_different_window_sizes()
        
        # 测试3: 多股票场景
        test_multi_stock_scenario()
        
        print("\n=== 所有测试完成 ===")
        print("滚动窗口归一化已成功集成到数据处理流程中")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()