import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

class RollingWindowScaler:
    """
    滚动窗口归一化器，适用于多股票时间序列数据
    支持不同的归一化方法和窗口策略
    """
    
    def __init__(self, window_size=252, method='zscore', min_periods=60):
        """
        Args:
            window_size (int): 滚动窗口大小，默认252个交易日（约1年）
            method (str): 归一化方法 ['zscore', 'minmax', 'robust']
            min_periods (int): 最小样本数，少于此数量时使用全局统计
        """
        self.window_size = window_size
        self.method = method
        self.min_periods = min_periods
        self.global_stats = {}  # 存储全局统计量作为备选
        
    def fit_transform(self, df, feature_columns):
        """
        对DataFrame进行滚动窗口归一化
        
        Args:
            df (pd.DataFrame): 包含时间序列数据的DataFrame
            feature_columns (list): 需要归一化的特征列名
            
        Returns:
            pd.DataFrame: 归一化后的DataFrame
        """
        df_normalized = df.copy()
        
        print(f"开始滚动窗口归一化，窗口大小: {self.window_size}, 方法: {self.method}")
        
        # 计算全局统计量作为备选
        self._calculate_global_stats(df, feature_columns)
        
        for col in feature_columns:
            if col not in df.columns:
                print(f"警告: 列 '{col}' 不存在，跳过")
                continue
                
            print(f"  处理特征: {col}")
            df_normalized[col] = self._normalize_column(df[col])
            
        return df_normalized
    
    def transform(self, df, feature_columns=None):
        """
        兼容sklearn scaler的transform方法
        对新数据使用已有的全局统计量进行归一化
        注意：这个方法只能在fit_transform之后使用
        """
        if not self.global_stats:
            raise ValueError("必须先调用fit_transform方法来计算统计量")
        
        if feature_columns is None:
            feature_columns = list(self.global_stats.keys())
        
        df_transformed = df.copy()
        
        for col in feature_columns:
            if col in df.columns and col in self.global_stats:
                df_transformed[col] = df_transformed[col].apply(
                    lambda x: self._normalize_with_global_stats(x, col)
                )
        
        return df_transformed[feature_columns]
    
    def _calculate_global_stats(self, df, feature_columns):
        """计算全局统计量"""
        for col in feature_columns:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    self.global_stats[col] = {
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max(),
                        'q25': data.quantile(0.25),
                        'q75': data.quantile(0.75),
                        'median': data.median()
                    }
    
    def _normalize_column(self, series):
        """对单列进行滚动窗口归一化（向量化优化版）"""
        # 确保输出是float类型，避免数据类型警告
        normalized_series = series.astype(float).copy()
        
        if self.method == 'zscore':
            # 使用pandas的rolling函数进行向量化计算
            rolling_mean = series.rolling(window=self.window_size, min_periods=self.min_periods).mean()
            rolling_std = series.rolling(window=self.window_size, min_periods=self.min_periods).std()
            
            # 向量化计算Z-score
            normalized_series = (series - rolling_mean) / rolling_std
            
            # 处理标准差为0的情况
            normalized_series = normalized_series.fillna(0)
            
            # 对于样本不足的早期数据，使用全局统计量
            insufficient_mask = rolling_std.isna()
            if insufficient_mask.any() and series.name in self.global_stats:
                global_mean = self.global_stats[series.name]['mean']
                global_std = self.global_stats[series.name]['std']
                if global_std > 0:
                    normalized_series[insufficient_mask] = (series[insufficient_mask] - global_mean) / global_std
                else:
                    normalized_series[insufficient_mask] = 0
                    
        elif self.method == 'minmax':
            # 向量化MinMax归一化
            rolling_min = series.rolling(window=self.window_size, min_periods=self.min_periods).min()
            rolling_max = series.rolling(window=self.window_size, min_periods=self.min_periods).max()
            
            range_vals = rolling_max - rolling_min
            normalized_series = (series - rolling_min) / range_vals
            
            # 处理范围为0的情况
            normalized_series = normalized_series.fillna(0.5)
            
            # 处理样本不足的情况
            insufficient_mask = rolling_min.isna()
            if insufficient_mask.any() and series.name in self.global_stats:
                global_min = self.global_stats[series.name]['min']
                global_max = self.global_stats[series.name]['max']
                if global_max > global_min:
                    normalized_series[insufficient_mask] = (series[insufficient_mask] - global_min) / (global_max - global_min)
                else:
                    normalized_series[insufficient_mask] = 0.5
                    
        elif self.method == 'robust':
            # 向量化Robust归一化
            rolling_median = series.rolling(window=self.window_size, min_periods=self.min_periods).median()
            rolling_q25 = series.rolling(window=self.window_size, min_periods=self.min_periods).quantile(0.25)
            rolling_q75 = series.rolling(window=self.window_size, min_periods=self.min_periods).quantile(0.75)
            
            iqr = rolling_q75 - rolling_q25
            normalized_series = (series - rolling_median) / iqr
            
            # 处理IQR为0的情况
            normalized_series = normalized_series.fillna(0)
            
            # 处理样本不足的情况
            insufficient_mask = rolling_median.isna()
            if insufficient_mask.any() and series.name in self.global_stats:
                global_median = self.global_stats[series.name]['median']
                global_iqr = self.global_stats[series.name]['q75'] - self.global_stats[series.name]['q25']
                if global_iqr > 0:
                    normalized_series[insufficient_mask] = (series[insufficient_mask] - global_median) / global_iqr
                else:
                    normalized_series[insufficient_mask] = 0
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
        
        return normalized_series
    
    def _normalize_with_window_stats(self, value, window_data):
        """使用窗口统计量进行归一化"""
        if pd.isna(value):
            return value
            
        if self.method == 'zscore':
            mean = window_data.mean()
            std = window_data.std()
            if std == 0:
                return 0.0
            return (value - mean) / std
            
        elif self.method == 'minmax':
            min_val = window_data.min()
            max_val = window_data.max()
            if max_val == min_val:
                return 0.5  # 中间值
            return (value - min_val) / (max_val - min_val)
            
        elif self.method == 'robust':
            median = window_data.median()
            q25 = window_data.quantile(0.25)
            q75 = window_data.quantile(0.75)
            iqr = q75 - q25
            if iqr == 0:
                return 0.0
            return (value - median) / iqr
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
    
    def _normalize_with_global_stats(self, value, column_name):
        """使用全局统计量进行归一化"""
        if pd.isna(value) or column_name not in self.global_stats:
            return value
            
        stats = self.global_stats[column_name]
        
        if self.method == 'zscore':
            if stats['std'] == 0:
                return 0.0
            return (value - stats['mean']) / stats['std']
            
        elif self.method == 'minmax':
            if stats['max'] == stats['min']:
                return 0.5
            return (value - stats['min']) / (stats['max'] - stats['min'])
            
        elif self.method == 'robust':
            iqr = stats['q75'] - stats['q25']
            if iqr == 0:
                return 0.0
            return (value - stats['median']) / iqr
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")

class MultiStockRollingScaler:
    """
    多股票滚动窗口归一化器
    为每只股票维护独立的滚动统计量
    """
    
    def __init__(self, window_size=252, method='zscore', min_periods=60):
        self.window_size = window_size
        self.method = method
        self.min_periods = min_periods
        self.stock_scalers = {}
        
    def fit_transform_multi_stock(self, stock_data_dict, feature_columns):
        """
        对多只股票进行滚动窗口归一化
        
        Args:
            stock_data_dict (dict): {stock_code: DataFrame} 格式的股票数据
            feature_columns (list): 需要归一化的特征列名
            
        Returns:
            dict: {stock_code: normalized_DataFrame} 格式的归一化数据
        """
        normalized_data = {}
        
        print(f"开始多股票滚动窗口归一化，共 {len(stock_data_dict)} 只股票")
        
        for stock_code, df in stock_data_dict.items():
            print(f"\n处理股票: {stock_code}")
            
            # 为每只股票创建独立的归一化器
            scaler = RollingWindowScaler(
                window_size=self.window_size,
                method=self.method,
                min_periods=self.min_periods
            )
            
            # 归一化
            normalized_df = scaler.fit_transform(df, feature_columns)
            normalized_data[stock_code] = normalized_df
            self.stock_scalers[stock_code] = scaler
            
        return normalized_data
    
    def get_scaler_stats(self, stock_code=None):
        """获取归一化器的统计信息"""
        if stock_code:
            return self.stock_scalers.get(stock_code, {}).global_stats
        else:
            return {code: scaler.global_stats for code, scaler in self.stock_scalers.items()}

# 使用示例和测试函数
def test_rolling_scaler():
    """测试滚动窗口归一化器"""
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 模拟股票数据（包含趋势和波动性变化）
    trend = np.linspace(100, 200, 500)
    noise = np.random.normal(0, 10, 500)
    volatility_change = np.where(np.arange(500) > 250, 2.0, 1.0)  # 后半段波动性增加
    
    test_data = pd.DataFrame({
        'date': dates,
        'close': trend + noise * volatility_change,
        'volume': np.random.lognormal(10, 1, 500),
        'pct_chg': np.random.normal(0, 2, 500)
    })
    
    # 测试归一化
    scaler = RollingWindowScaler(window_size=60, method='zscore')
    normalized_data = scaler.fit_transform(test_data, ['close', 'volume', 'pct_chg'])
    
    print("原始数据统计:")
    print(test_data[['close', 'volume', 'pct_chg']].describe())
    print("\n归一化后数据统计:")
    print(normalized_data[['close', 'volume', 'pct_chg']].describe())
    
    return normalized_data

if __name__ == '__main__':
    test_rolling_scaler()