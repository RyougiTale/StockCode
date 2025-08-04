import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# 假设 stock_util 在项目根目录
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_stock_by_code

from . import config

def _prepare_and_scale_stock_data(df, feature_columns):
    """
    (内部函数) 对单只股票的DataFrame进行预处理和独立的特征归一化。
    """
    # 1. 特征工程
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA60'] = df['close'].rolling(window=60).mean()
    
    # 2. 清理数据
    clean_df = df[feature_columns].dropna().reset_index(drop=True)
    if len(clean_df) < config.MAX_SEQ_LEN + 1:
        return None, None, None

    # 3. 【修正】按特征独立归一化
    scalers = {}
    scaled_df = pd.DataFrame()
    for col in feature_columns:
        scaler = MinMaxScaler()
        scaled_df[col] = scaler.fit_transform(clean_df[[col]]).flatten()
        scalers[col] = scaler
    
    return scaled_df.values, scalers, clean_df.tail(config.MAX_SEQ_LEN)

def prepare_multistock_data(stock_codes):
    """
    加载、处理和准备多只股票的数据用于训练。
    """
    print(f"--- Loading, Processing and Preparing Data for {len(stock_codes)} stocks ---")
    
    all_sequences = []
    all_scalers = {} # 现在存储结构为 {stock_code: {feature: scaler}}
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
    final_feature_columns = feature_columns + ['SMA20', 'SMA60']

    for code in stock_codes:
        print(f"Processing {code}...")
        full_df = read_history_stock_by_code(code)
        if full_df is None or full_df.empty:
            print(f"Warning: Could not read data for stock code: {code}. Skipping.")
            continue
            
        scaled_data, scalers, _ = _prepare_and_scale_stock_data(full_df, final_feature_columns)
        
        if scaled_data is not None:
            # 为这只股票创建序列
            for i in range(len(scaled_data) - config.MAX_SEQ_LEN):
                all_sequences.append(scaled_data[i:i + config.MAX_SEQ_LEN + 1])
            all_scalers[code] = scalers
        else:
            print(f"Warning: Not enough data for {code} after processing. Skipping.")

    if not all_sequences:
        print("Error: No data available for any of the provided stock codes.")
        return None, None, None

    combined_sequences = np.array(all_sequences)
    
    print("--- Multi-stock Data Ready ---")
    return combined_sequences, all_scalers, final_feature_columns

def prepare_singlestock_data_for_inference(stock_code):
    """
    为单只股票准备用于推理的数据。
    """
    print(f"--- Preparing single stock data for inference: {stock_code} ---")
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
    final_feature_columns = feature_columns + ['SMA20', 'SMA60']

    full_df = read_history_stock_by_code(stock_code)
    if full_df is None or full_df.empty:
        print(f"Error: Could not read data for stock code: {stock_code}.")
        return None, None, None, None

    scaled_data, scalers, inference_df = _prepare_and_scale_stock_data(full_df.copy(), final_feature_columns)

    if scalers is None:
        print(f"Error: Not enough data for {stock_code} to create inference sequence.")
        return None, None, None, None

    # 我们只需要最后一段用于输入的数据
    inference_input_data = scaled_data[-config.MAX_SEQ_LEN:]

    return inference_input_data, scalers, final_feature_columns, inference_df


class StockDataset(Dataset):
    """
    为Transformer模型创建序列到序列的数据集。
    """
    def __init__(self, sequences_array):
        self.sequences = sequences_array

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_seq = self.sequences[idx]
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)