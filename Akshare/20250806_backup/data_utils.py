import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# 利用项目根目录下的工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

from . import config

def resample_to_period(df, period='W-FRI'):
    """将日K数据降采样为周K或月K。"""
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

def calculate_features(df, period):
    """根据不同的时间尺度计算相应的技术指标"""
    indicators_to_calc = config.TECH_INDICATORS.get(period, [])
    
    if 'SMA20' in indicators_to_calc:
        df['SMA20'] = df['close'].rolling(window=20).mean()
    if 'SMA60' in indicators_to_calc:
        df['SMA60'] = df['close'].rolling(window=60).mean()
        
    return df

def calculate_future_metrics(price_series):
    """
    计算未来N天窗口内的四个核心指标。
    Args:
        price_series (pd.Series): 未来N天的价格序列。
    Returns:
        dict: 包含四个指标的字典，如果数据不足则返回None。
    """
    if len(price_series) < 2:
        return None

    final_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    cumulative_max = price_series.cummax()
    drawdown = (price_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    daily_returns = price_series.pct_change().dropna()
    volatility = daily_returns.std()

    return {
        "final_return": final_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }

# (classify_market_pattern 函数将被移除)

def create_samples_for_code(code):
    """为单只股票/指数创建所有样本（新版：软标签）"""
    print(f"Processing data for {code}...")
    daily_df = read_history_by_code(code)
    if daily_df is None or daily_df.empty:
        return [], {}

    # --- 步骤 1: 计算未来回报率 (标签) ---
    look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    daily_df['label'] = daily_df['close'].shift(-look_forward_days) / daily_df['close'] - 1
    
    # --- 步骤 2: 计算输入特征 (分时间尺度) ---
    daily_df = calculate_features(daily_df, 'daily')
    
    # --- 步骤 3: 准备多时间尺度数据 ---
    weekly_df = resample_to_period(daily_df.copy(), 'W-FRI')
    weekly_df = calculate_features(weekly_df, 'weekly')
    
    monthly_df = resample_to_period(daily_df.copy(), 'ME')
    monthly_df = calculate_features(monthly_df, 'monthly')

    # --- 步骤 4: 数据归一化 ---
    print("\n--- Data before normalization ---")
    print(f"Daily df length: {len(daily_df)}")
    print("Daily Head before norm:\n", daily_df.head())
    print("Daily tail before norm:\n", daily_df.tail())
    print("Daily NaN counts:\n", daily_df[config.FEATURE_COLUMNS['daily']].isnull().sum())
    
    print(f"\nWeekly df length: {len(weekly_df)}")
    print("Weekly Head before norm:\n", weekly_df.head())
    print("Weekly tail before norm:\n", weekly_df.tail())
    print("Weekly NaN counts:\n", weekly_df[config.FEATURE_COLUMNS['weekly']].isnull().sum())
    
    print(f"\nMonthly df length: {len(monthly_df)}")
    print("Monthly Head before norm:\n", monthly_df.head())
    print("Monthly tail before norm:\n", monthly_df.tail())
    print("Monthly NaN counts:\n", monthly_df[config.FEATURE_COLUMNS['monthly']].isnull().sum())

    daily_scaler = MinMaxScaler()
    daily_df[config.FEATURE_COLUMNS['daily']] = daily_scaler.fit_transform(daily_df[config.FEATURE_COLUMNS['daily']])
    
    weekly_scaler = MinMaxScaler()
    weekly_df[config.FEATURE_COLUMNS['weekly']] = weekly_scaler.fit_transform(weekly_df[config.FEATURE_COLUMNS['weekly']])

    monthly_scaler = MinMaxScaler()
    monthly_df[config.FEATURE_COLUMNS['monthly']] = monthly_scaler.fit_transform(monthly_df[config.FEATURE_COLUMNS['monthly']])

    print("\n--- Data after normalization (showing head) ---")
    print("Daily Head:\n", daily_df.head())
    print("Weekly Head:\n", weekly_df.head())
    print("Monthly Head:\n", monthly_df.head())

    # --- 步骤 5: 创建样本 ---
    samples = []
    daily_df.dropna(subset=['label'] + config.FEATURE_COLUMNS['daily'], inplace=True)

    for i in range(len(daily_df) - 1, config.DAILY_SEQ_LEN - 1, -1):
        current_date = daily_df.iloc[i]['date']
        
        daily_end_idx = i + 1
        daily_start_idx = daily_end_idx - config.DAILY_SEQ_LEN
        daily_slice = daily_df.iloc[daily_start_idx:daily_end_idx]
        
        weekly_slice = weekly_df[weekly_df['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        monthly_slice = monthly_df[monthly_df['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        
        if (len(daily_slice) == config.DAILY_SEQ_LEN and
            len(weekly_slice) == config.WEEKLY_SEQ_LEN and
            len(monthly_slice) == config.MONTHLY_SEQ_LEN):
            
            daily_data = daily_slice[config.FEATURE_COLUMNS['daily']].values
            weekly_data = weekly_slice[config.FEATURE_COLUMNS['weekly']].values
            monthly_data = monthly_slice[config.FEATURE_COLUMNS['monthly']].values
            label = daily_df.iloc[i]['label']
            future_prices = daily_df['close'].iloc[i : i + look_forward_days].values

            if np.isnan(daily_data).any() or np.isnan(weekly_data).any() or np.isnan(monthly_data).any() or pd.isna(label):
                continue
            
            sample = {
                'date': current_date,
                'daily': daily_data,
                'weekly': weekly_data,
                'monthly': monthly_data,
                'label': label,
                'future_prices': future_prices
            }
            samples.append(sample)
    
    # 返回样本和拟合好的scalers
    scalers = {'daily': daily_scaler, 'weekly': weekly_scaler, 'monthly': monthly_scaler}
    return samples[::-1], scalers

def get_all_samples(stock_codes):
    """获取所有股票代码的样本，并返回第一个代码的scalers"""
    all_samples = []
    all_scalers = {} # 保存每个代码的scalers
    
    for code in stock_codes:
        samples, scalers = create_samples_for_code(code)
        if samples: # 只有在成功创建样本时才添加
            all_samples.extend(samples)
            if not all_scalers: # 只保存第一个代码的scalers作为代表
                all_scalers = scalers
                
    # 注意：这里假设所有股票的数据分布相似，因此只使用第一只股票的scaler。
    # 在更复杂的场景中，可能需要为每个股票单独保存或使用一个全局scaler。
    return all_samples, all_scalers