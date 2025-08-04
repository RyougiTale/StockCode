import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# 利用项目根目录下的工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_stock_by_code

from . import config

def resample_to_period(df, period='W'):
    """将日K数据降采样为周K或月K"""
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    logic = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'turnover': 'sum',
        'amplitude': lambda x: (x.max() - x.min()) / x.iloc[0] * 100 if not x.empty and x.iloc[0] != 0 else 0,
        'pct_chg': lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if not x.empty and x.iloc[0] != 0 else 0,
        'chg_amount': 'sum',
        'turnover_rate': 'sum'
    }
    
    resampled_df = df.resample(period).apply(logic).dropna()
    return resampled_df.reset_index()

def calculate_features(df):
    """计算技术指标"""
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA60'] = df['close'].rolling(window=60).mean()
    return df

def calculate_labels(df, look_forward_days):
    """计算标签：未来N日收盘价是否上涨"""
    df['future_price'] = df['close'].shift(-look_forward_days)
    df['label'] = (df['future_price'] > df['close']).astype(int)
    return df

def create_samples_for_code(code):
    """为单只股票/指数创建所有样本"""
    print(f"Processing data for {code}...")
    daily_df = read_history_stock_by_code(code)
    if daily_df is None or daily_df.empty:
        return []

    # 1. 数据准备和特征计算
    daily_df = calculate_features(daily_df)
    daily_df = calculate_labels(daily_df, config.LABEL_LOOK_FORWARD_DAYS)
    
    weekly_df = resample_to_period(daily_df.copy(), 'W')
    weekly_df = calculate_features(weekly_df)
    
    monthly_df = resample_to_period(daily_df.copy(), 'M')
    monthly_df = calculate_features(monthly_df)

    # 【调试】为周K和月K添加一个假的 'label' 列以匹配特征维度
    weekly_df['label'] = 0
    monthly_df['label'] = 0

    # 2. 数据归一化（按特征独立归一化）
    feature_cols = config.FEATURE_COLUMNS
    daily_scaler = MinMaxScaler()
    daily_df[feature_cols] = daily_scaler.fit_transform(daily_df[feature_cols])
    
    weekly_scaler = MinMaxScaler()
    weekly_df[feature_cols] = weekly_scaler.fit_transform(weekly_df[feature_cols])

    monthly_scaler = MinMaxScaler()
    monthly_df[feature_cols] = monthly_scaler.fit_transform(monthly_df[feature_cols])

    # 3. 创建样本
    samples = []
    # 我们以日K为基准，从后往前遍历，确保有足够的历史数据
    # 至少需要60天日K, 52周周K, 24个月月K
    for i in range(len(daily_df) - 1, config.DAILY_SEQ_LEN - 1, -1):
        current_date = daily_df.loc[i, 'date']
        
        # 获取日K序列
        daily_end_idx = i + 1
        daily_start_idx = daily_end_idx - config.DAILY_SEQ_LEN
        daily_slice = daily_df.iloc[daily_start_idx:daily_end_idx]
        
        # 获取周K序列
        weekly_slice = weekly_df[weekly_df['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        
        # 获取月K序列
        monthly_slice = monthly_df[monthly_df['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        
        # 检查所有序列长度是否足够
        if (len(daily_slice) == config.DAILY_SEQ_LEN and
            len(weekly_slice) == config.WEEKLY_SEQ_LEN and
            len(monthly_slice) == config.MONTHLY_SEQ_LEN):
            
            # 最后的数据质量检查
            daily_data = daily_slice[feature_cols].values
            weekly_data = weekly_slice[feature_cols].values
            monthly_data = monthly_slice[feature_cols].values
            label = daily_df.loc[i, 'label']

            if (np.isnan(daily_data).any() or np.isnan(weekly_data).any() or
                np.isnan(monthly_data).any() or np.isnan(label)):
                # print("contain nan...")
                # 如果任何数据中包含NaN，则跳过这个样本
                continue
            
            sample = {
                'daily': daily_data,
                'weekly': weekly_data,
                'monthly': monthly_data,
                'label': label
            }
            samples.append(sample)
    
    # 因为我们是倒序遍历的，所以最后要把样本反转过来，保持时间顺序
    return samples[::-1]

def get_all_samples(stock_codes):
    """获取所有股票代码的样本"""
    all_samples = []
    for code in stock_codes:
        samples = create_samples_for_code(code)
        all_samples.extend(samples)
    return all_samples