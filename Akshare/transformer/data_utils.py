import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# 假设 stock_util 在项目根目录
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_stock_by_code

from . import config

# =============================================================================
# 1. 数据处理核心函数
# =============================================================================
def prepare_data(stock_code):
    """
    加载、处理和准备指定股票代码的数据。

    Args:
        stock_code (str): 股票代码。

    Returns:
        tuple: 包含训练数据、推断输入数据、缩放器、特征列和推断DataFrame。
               如果数据不足，则返回 (None, None, None, None, None)。
    """
    print(f"--- Loading, Processing and Preparing Data for {stock_code} ---")
    full_df = read_history_stock_by_code(stock_code)
    if full_df is None or full_df.empty:
        print(f"Could not read data for stock code: {stock_code}"); return None, None, None, None, None

    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
    
    # 计算技术指标
    full_df['SMA20'] = full_df['close'].rolling(window=20).mean()
    full_df['SMA60'] = full_df['close'].rolling(window=60).mean()
    feature_columns.extend(['SMA20', 'SMA60'])
    
    # 清理包含NaN的数据
    clean_df = full_df[feature_columns].dropna().reset_index(drop=True)
    
    if len(clean_df) < config.MAX_SEQ_LEN * 2:
        print("Not enough data after cleaning."); return None, None, None, None, None

    # 划分训练集和推断集
    train_df = clean_df[:-config.MAX_SEQ_LEN]
    inference_df = clean_df.tail(config.MAX_SEQ_LEN).reset_index(drop=True)

    # 数据归一化
    scalers = {}
    train_scaled_df = pd.DataFrame()
    for col in feature_columns:
        scaler = MinMaxScaler()
        train_scaled_df[col] = scaler.fit_transform(train_df[[col]]).flatten()
        scalers[col] = scaler
    
    # 使用训练集的scaler来转换推理数据
    inference_scaled_df = pd.DataFrame()
    for col in feature_columns:
        inference_scaled_df[col] = scalers[col].transform(inference_df[[col]]).flatten()

    train_data = train_scaled_df.values
    inference_input_data = inference_scaled_df.values
    
    print("--- Data Ready ---")
    return train_data, inference_input_data, scalers, feature_columns, inference_df

# =============================================================================
# 2. PyTorch 数据集类
# =============================================================================
class StockDataset(Dataset):
    """
    为Transformer模型创建序列到序列的数据集。
    """
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # 确保我们有足够的数据来创建一个完整的输入-目标对
        return len(self.sequences) - config.MAX_SEQ_LEN

    def __getitem__(self, idx):
        # 输入序列是 [idx, idx + MAX_SEQ_LEN)
        input_seq = self.sequences[idx : idx + config.MAX_SEQ_LEN]
        # 目标序列是 [idx + 1, idx + MAX_SEQ_LEN + 1)
        target_seq = self.sequences[idx + 1 : idx + config.MAX_SEQ_LEN + 1]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)