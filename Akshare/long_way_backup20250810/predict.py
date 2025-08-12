import torch
import pandas as pd
import numpy as np
import joblib
import sys
import os

# --- 路径设置，确保能找到项目根目录下的模块 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

# --- 导入 long_way 模块 ---
from . import config
from .model import MultiEncoderFusionModel
from .data_utils import resample_to_period, calculate_features

def predict_for_date(stock_code, target_date_str):
    """
    为一个给定的股票和日期，加载模型并预测未来的回报率分布。
    
    Args:
        stock_code (str): 股票代码, e.g., "600036"
        target_date_str (str): 您希望进行预测的“今天”的日期, e.g., "2023-12-31"。
                             模型将使用这一天及之前的所有数据，来预测未来20个交易日的情况。
    """
    print(f"--- Starting prediction for {stock_code} on {target_date_str} ---")

    # --- 1. 加载模型和Scalers ---
    print("Loading model and scalers...")
    # 初始化模型结构
    daily_config = {
        'feature_size': len(config.FEATURE_COLUMNS['daily']),
        **config.SHARED_ENCODER_CONFIG
    }
    weekly_config = {
        'feature_size': len(config.FEATURE_COLUMNS['weekly']),
        **config.SHARED_ENCODER_CONFIG
    }
    monthly_config = {
        'feature_size': len(config.FEATURE_COLUMNS['monthly']),
        **config.SHARED_ENCODER_CONFIG
    }
    model = MultiEncoderFusionModel(
        daily_config=daily_config,
        weekly_config=weekly_config,
        monthly_config=monthly_config,
        fusion_dim=config.FUSION_DIM,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)

    # 加载训练好的权重
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    print(f"Model loaded from {config.MODEL_PATH}")

    # 注意：滚动窗口归一化不需要加载保存的scaler
    # 而是在预测时重新创建并使用最新数据进行归一化
    print("使用滚动窗口归一化，将基于最新数据动态计算...")

    # --- 2. 准备数据 ---
    print("Preparing data...")
    full_daily_df = read_history_by_code(stock_code)
    if full_daily_df is None or full_daily_df.empty:
        print(f"Could not read data for {stock_code}")
        return

    target_date = pd.to_datetime(target_date_str)
    daily_df = full_daily_df[full_daily_df['date'] <= target_date].copy()

    if len(daily_df) < config.DAILY_SEQ_LEN:
        print(f"Error: Not enough historical data. Need at least {config.DAILY_SEQ_LEN} days, but got {len(daily_df)}.")
        return

    # a. 计算特征
    daily_df_featured = calculate_features(daily_df)
    weekly_df_featured = calculate_features(resample_to_period(daily_df.copy(), 'W'))
    monthly_df_featured = calculate_features(resample_to_period(daily_df.copy(), 'M'))

    # b. 数据归一化 (使用滚动窗口归一化，基于最新数据)
    from .rolling_scaler import RollingWindowScaler
    
    # 为预测创建新的滚动窗口归一化器
    daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
    monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)
    
    # 使用完整的历史数据进行归一化（这样预测时的归一化是基于最新的数据分布）
    daily_df_featured = daily_scaler.fit_transform(daily_df_featured, config.FEATURE_COLUMNS['daily'])
    weekly_df_featured = weekly_scaler.fit_transform(weekly_df_featured, config.FEATURE_COLUMNS['weekly'])
    monthly_df_featured = monthly_scaler.fit_transform(monthly_df_featured, config.FEATURE_COLUMNS['monthly'])
    
    print("数据归一化完成（基于最新数据窗口）")

    # c. 提取最终的序列
    daily_slice = daily_df_featured.tail(config.DAILY_SEQ_LEN)
    weekly_slice = weekly_df_featured.tail(config.WEEKLY_SEQ_LEN)
    monthly_slice = monthly_df_featured.tail(config.MONTHLY_SEQ_LEN)

    if len(daily_slice) < config.DAILY_SEQ_LEN or len(weekly_slice) < config.WEEKLY_SEQ_LEN or len(monthly_slice) < config.MONTHLY_SEQ_LEN:
        print("Error: Not enough data to form complete sequences after resampling.")
        return

    # d. 转换为Tensor
    daily_tensor = torch.from_numpy(daily_slice[config.FEATURE_COLUMNS].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
    weekly_tensor = torch.from_numpy(weekly_slice[config.FEATURE_COLUMNS].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
    monthly_tensor = torch.from_numpy(monthly_slice[config.FEATURE_COLUMNS].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
    print("Data prepared successfully.")

    # --- 3. 执行预测 ---
    print("Running prediction...")
    with torch.no_grad():
        log_probs = model(daily_tensor, weekly_tensor, monthly_tensor)
        probabilities = torch.exp(log_probs).squeeze(0)

    # --- 4. 展示结果 ---
    print("\n--- Prediction Result ---")
    print(f"Stock: {stock_code}")
    print(f"Date: {target_date_str}")
    print(f"Predicted probability distribution for the next {config.SOFT_LABEL_CONFIG['LOOK_FORWARD_DAYS']} trading days:")
    
    centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
    probs = probabilities.cpu().numpy()

    for i, center in enumerate(centers):
        print(f"  - Return Center {center*100: >5.1f}%: Probability = {probs[i]:.4f}")
    
    print("-------------------------\n")

if __name__ == '__main__':
    # --- 在这里修改你要预测的股票和日期 ---
    STOCK_TO_PREDICT = "600036"
    DATE_TO_PREDICT = "2025-03-20" # 您想在哪一天进行预测？脚本将使用这一天及之前的数据。

    predict_for_date(STOCK_TO_PREDICT, DATE_TO_PREDICT)