import torch
import pandas as pd
import numpy as np
import joblib
import sys
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 路径设置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

# --- 导入 long_way 模块 ---
from . import config
from .model import MultiEncoderFusionModel
from .data_utils import resample_to_period, calculate_features

def draw_prediction_vs_actual(stock_code, years=5, model_path=None, ax=None):
    """
    为单个模型绘制预测与真实的对比图。
    """
    if ax is None:
        fig, ax_local = plt.subplots(figsize=(15, 7))
    else:
        ax_local = ax

    model_to_load = model_path if model_path else config.MODEL_PATH
    model_name = os.path.basename(model_to_load).replace('.pth', '')
    
    daily_config = {'feature_size': len(config.FEATURE_COLUMNS['daily']), **config.SHARED_ENCODER_CONFIG}
    weekly_config = {'feature_size': len(config.FEATURE_COLUMNS['weekly']), **config.SHARED_ENCODER_CONFIG}
    monthly_config = {'feature_size': len(config.FEATURE_COLUMNS['monthly']), **config.SHARED_ENCODER_CONFIG}
    
    model = MultiEncoderFusionModel(
        daily_config=daily_config, weekly_config=weekly_config, monthly_config=monthly_config,
        fusion_dim=config.FUSION_DIM, num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(model_to_load, map_location=config.DEVICE))
    model.eval()

    # 使用滚动窗口归一化，基于最新数据动态计算
    from .rolling_scaler import RollingWindowScaler
    
    daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
    monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)

    full_daily_df = read_history_by_code(stock_code)
    look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    full_daily_df['actual_return'] = full_daily_df['close'].shift(-look_forward_days) / full_daily_df['close'] - 1

    daily_featured = calculate_features(full_daily_df.copy(), 'daily')
    weekly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'W-FRI'), 'weekly')
    monthly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'ME'), 'monthly')

    # 使用完整历史数据进行滚动窗口归一化
    daily_featured = daily_scaler.fit_transform(daily_featured, config.FEATURE_COLUMNS['daily'])
    weekly_featured = weekly_scaler.fit_transform(weekly_featured, config.FEATURE_COLUMNS['weekly'])
    monthly_featured = monthly_scaler.fit_transform(monthly_featured, config.FEATURE_COLUMNS['monthly'])
    
    end_date = full_daily_df['date'].max()
    start_date = end_date - pd.DateOffset(years=years)
    target_df = full_daily_df[full_daily_df['date'] >= start_date].copy()
    
    results = []
    for index, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"Predicting for {model_name}"):
        current_date = row['date']
        daily_slice = daily_featured[daily_featured['date'] <= current_date].tail(config.DAILY_SEQ_LEN)
        weekly_slice = weekly_featured[weekly_featured['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        monthly_slice = monthly_featured[monthly_featured['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        if not (len(daily_slice) == config.DAILY_SEQ_LEN and len(weekly_slice) == config.WEEKLY_SEQ_LEN and len(monthly_slice) == config.MONTHLY_SEQ_LEN): continue
        daily_tensor = torch.from_numpy(daily_slice[config.FEATURE_COLUMNS['daily']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        weekly_tensor = torch.from_numpy(weekly_slice[config.FEATURE_COLUMNS['weekly']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        monthly_tensor = torch.from_numpy(monthly_slice[config.FEATURE_COLUMNS['monthly']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            log_probs = model(daily_tensor, weekly_tensor, monthly_tensor)
            probabilities = torch.exp(log_probs).squeeze(0).cpu().numpy()
        centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
        
        expected_return_full = np.sum(probabilities * centers)
        
        top3_indices = np.argsort(probabilities)[-3:]; top3_probs = probabilities[top3_indices]; top3_centers = centers[top3_indices]
        expected_return_top3 = np.sum((top3_probs / np.sum(top3_probs)) * top3_centers)
        
        top2_indices = np.argsort(probabilities)[-2:]; top2_probs = probabilities[top2_indices]; top2_centers = centers[top2_indices]
        expected_return_top2 = np.sum((top2_probs / np.sum(top2_probs)) * top2_centers)

        top1_index = np.argmax(probabilities)
        expected_return_top1 = centers[top1_index]

        results.append({'date': current_date, 'predicted_return': expected_return_full, 'predicted_return_top3': expected_return_top3, 'predicted_return_top2': expected_return_top2, 'predicted_return_top1': expected_return_top1, 'actual_return': row['actual_return']})

    result_df = pd.DataFrame(results)
    historical_df = result_df.dropna()
    
    sns.lineplot(x='date', y='predicted_return', data=result_df, ax=ax_local, label='Pred. (Full)', color='royalblue', alpha=0.6)
    sns.lineplot(x='date', y='predicted_return_top1', data=result_df, ax=ax_local, label='Pred. (Top-1)', color='red', linewidth=2, linestyle=':')
    sns.lineplot(x='date', y='predicted_return_top2', data=result_df, ax=ax_local, label='Pred. (Top-2)', color='mediumseagreen', linewidth=2, linestyle='--')
    sns.lineplot(x='date', y='predicted_return_top3', data=result_df, ax=ax_local, label='Pred. (Top-3)', color='darkviolet', linewidth=2)
    sns.lineplot(x='date', y='actual_return', data=historical_df, ax=ax_local, label='Actual', color='darkorange', alpha=0.8)
    
    if not historical_df.empty:
        ax_local.axvline(x=historical_df['date'].max(), color='red', linestyle='--', linewidth=1.5, label='Forecast Start')

    ax_local.set_title(f'Performance of: {model_name}', fontsize=14)
    ax_local.set_ylabel('Return')
    ax_local.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    ax_local.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax_local.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()

def draw_all_best_models(stock_code, years=1):
    """
    优化版：只计算一次数据预处理，然后为所有模型绘制预测。
    """
    model_paths = glob.glob(os.path.join(config.MODEL_DIR, "model_best_*.pth"))
    if not model_paths:
        print("No best models found. Running single prediction for default model.")
        draw_prediction_vs_actual(stock_code, years)
        return

    print(f"Found {len(model_paths)} best models to draw.")
    
    # === 只计算一次数据预处理 ===
    print("预处理数据（只计算一次）...")
    from .rolling_scaler import RollingWindowScaler
    
    daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
    monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)

    full_daily_df = read_history_by_code(stock_code)
    look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    full_daily_df['actual_return'] = full_daily_df['close'].shift(-look_forward_days) / full_daily_df['close'] - 1

    daily_featured = calculate_features(full_daily_df.copy(), 'daily')
    weekly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'W-FRI'), 'weekly')
    monthly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'ME'), 'monthly')

    # 使用完整历史数据进行滚动窗口归一化（只计算一次）
    daily_featured = daily_scaler.fit_transform(daily_featured, config.FEATURE_COLUMNS['daily'])
    weekly_featured = weekly_scaler.fit_transform(weekly_featured, config.FEATURE_COLUMNS['weekly'])
    monthly_featured = monthly_scaler.fit_transform(monthly_featured, config.FEATURE_COLUMNS['monthly'])
    
    end_date = full_daily_df['date'].max()
    start_date = end_date - pd.DateOffset(years=years)
    target_df = full_daily_df[full_daily_df['date'] >= start_date].copy()
    
    print("数据预处理完成，开始为所有模型生成预测...")
    
    num_models = len(model_paths)
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
    axes = axes.flatten()

    for i, model_path in enumerate(sorted(model_paths)):
        if i < len(axes):
            # 使用预处理好的数据绘制单个模型
            draw_single_model_with_preprocessed_data(
                model_path, target_df, daily_featured, weekly_featured, monthly_featured, axes[i]
            )

    # 隐藏多余的子图
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Comparison of Best Models for {stock_code}', fontsize=20, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = os.path.join(config.MODEL_DIR, f"best_models_comparison_{stock_code}.png")
    plt.savefig(output_path)
    print(f"\nComparison plot saved to {output_path}")
    plt.show()

def draw_single_model_with_preprocessed_data(model_path, target_df, daily_featured, weekly_featured, monthly_featured, ax):
    """
    使用预处理好的数据为单个模型绘制预测图
    """
    model_name = os.path.basename(model_path).replace('.pth', '')
    
    # 加载模型
    daily_config = {'feature_size': len(config.FEATURE_COLUMNS['daily']), **config.SHARED_ENCODER_CONFIG}
    weekly_config = {'feature_size': len(config.FEATURE_COLUMNS['weekly']), **config.SHARED_ENCODER_CONFIG}
    monthly_config = {'feature_size': len(config.FEATURE_COLUMNS['monthly']), **config.SHARED_ENCODER_CONFIG}
    
    model = MultiEncoderFusionModel(
        daily_config=daily_config, weekly_config=weekly_config, monthly_config=monthly_config,
        fusion_dim=config.FUSION_DIM, num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # 生成预测
    results = []
    for index, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"Predicting for {model_name}"):
        current_date = row['date']
        daily_slice = daily_featured[daily_featured['date'] <= current_date].tail(config.DAILY_SEQ_LEN)
        weekly_slice = weekly_featured[weekly_featured['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        monthly_slice = monthly_featured[monthly_featured['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        
        if not (len(daily_slice) == config.DAILY_SEQ_LEN and len(weekly_slice) == config.WEEKLY_SEQ_LEN and len(monthly_slice) == config.MONTHLY_SEQ_LEN):
            continue
            
        daily_tensor = torch.from_numpy(daily_slice[config.FEATURE_COLUMNS['daily']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        weekly_tensor = torch.from_numpy(weekly_slice[config.FEATURE_COLUMNS['weekly']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        monthly_tensor = torch.from_numpy(monthly_slice[config.FEATURE_COLUMNS['monthly']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            log_probs = model(daily_tensor, weekly_tensor, monthly_tensor)
            probabilities = torch.exp(log_probs).squeeze(0).cpu().numpy()
            
        centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
        
        expected_return_full = np.sum(probabilities * centers)
        
        top3_indices = np.argsort(probabilities)[-3:]; top3_probs = probabilities[top3_indices]; top3_centers = centers[top3_indices]
        expected_return_top3 = np.sum((top3_probs / np.sum(top3_probs)) * top3_centers)
        
        top2_indices = np.argsort(probabilities)[-2:]; top2_probs = probabilities[top2_indices]; top2_centers = centers[top2_indices]
        expected_return_top2 = np.sum((top2_probs / np.sum(top2_probs)) * top2_centers)

        top1_index = np.argmax(probabilities)
        expected_return_top1 = centers[top1_index]

        results.append({
            'date': current_date,
            'predicted_return': expected_return_full,
            'predicted_return_top3': expected_return_top3,
            'predicted_return_top2': expected_return_top2,
            'predicted_return_top1': expected_return_top1,
            'actual_return': row['actual_return']
        })

    result_df = pd.DataFrame(results)
    historical_df = result_df.dropna()
    
    # 绘制图表
    sns.lineplot(x='date', y='predicted_return', data=result_df, ax=ax, label='Pred. (Full)', color='royalblue', alpha=0.6)
    sns.lineplot(x='date', y='predicted_return_top1', data=result_df, ax=ax, label='Pred. (Top-1)', color='red', linewidth=2, linestyle=':')
    sns.lineplot(x='date', y='predicted_return_top2', data=result_df, ax=ax, label='Pred. (Top-2)', color='mediumseagreen', linewidth=2, linestyle='--')
    sns.lineplot(x='date', y='predicted_return_top3', data=result_df, ax=ax, label='Pred. (Top-3)', color='darkviolet', linewidth=2)
    sns.lineplot(x='date', y='actual_return', data=historical_df, ax=ax, label='Actual', color='darkorange', alpha=0.8)
    
    if not historical_df.empty:
        ax.axvline(x=historical_df['date'].max(), color='red', linestyle='--', linewidth=1.5, label='Forecast Start')

    ax.set_title(f'Performance of: {model_name}', fontsize=14)
    ax.set_ylabel('Return')
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.legend()

if __name__ == '__main__':
    STOCK_TO_DRAW = config.STOCK_CODES[0]
    YEARS_TO_DRAW = 3
    draw_all_best_models(STOCK_TO_DRAW, YEARS_TO_DRAW)