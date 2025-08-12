import torch
import pandas as pd
import numpy as np
import sys
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 路径设置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

# --- 导入 long_way 模块 ---
from . import config
from .model_3d import create_3d_model
from .data_utils import resample_to_period, calculate_features
from .label_3d_generator import ThreeDimensionalLabelGenerator

def draw_3d_simple(stock_code, years=2, model_path=None):
    """
    简化版3D可视化 - 多子图显示三个维度
    避免复杂依赖，专注核心功能
    """
    
    # 加载模型
    model_to_load = model_path if model_path else find_best_3d_model()
    if not model_to_load:
        print("未找到3D模型文件")
        return None
        
    model_name = os.path.basename(model_to_load).replace('.pth', '')
    print(f"加载3D模型: {model_name}")
    
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_to_load, map_location=config.DEVICE))
    model.eval()

    # 数据预处理
    from .rolling_scaler import RollingWindowScaler
    
    daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
    monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)

    full_daily_df = read_history_by_code(stock_code)
    look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    
    # 计算真实的三个指标
    label_generator = ThreeDimensionalLabelGenerator(look_forward_days=look_forward_days)
    
    daily_featured = calculate_features(full_daily_df.copy(), 'daily')
    weekly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'W-FRI'), 'weekly')
    monthly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'ME'), 'monthly')

    # 归一化
    daily_featured = daily_scaler.fit_transform(daily_featured, config.FEATURE_COLUMNS['daily'])
    weekly_featured = weekly_scaler.fit_transform(weekly_featured, config.FEATURE_COLUMNS['weekly'])
    monthly_featured = monthly_scaler.fit_transform(monthly_featured, config.FEATURE_COLUMNS['monthly'])
    
    end_date = full_daily_df['date'].max()
    start_date = end_date - pd.DateOffset(years=years)
    target_df = full_daily_df[full_daily_df['date'] >= start_date].copy()
    
    # 生成预测结果
    results = []
    for index, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"3D预测中"):
        current_date = row['date']
        
        # 获取输入数据
        daily_slice = daily_featured[daily_featured['date'] <= current_date].tail(config.DAILY_SEQ_LEN)
        weekly_slice = weekly_featured[weekly_featured['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        monthly_slice = monthly_featured[monthly_featured['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        
        if not (len(daily_slice) == config.DAILY_SEQ_LEN and 
                len(weekly_slice) == config.WEEKLY_SEQ_LEN and 
                len(monthly_slice) == config.MONTHLY_SEQ_LEN):
            continue
            
        # 模型预测
        daily_tensor = torch.from_numpy(daily_slice[config.FEATURE_COLUMNS['daily']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        weekly_tensor = torch.from_numpy(weekly_slice[config.FEATURE_COLUMNS['weekly']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        monthly_tensor = torch.from_numpy(monthly_slice[config.FEATURE_COLUMNS['monthly']].values.astype(np.float32)).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            outputs = model(daily_tensor, weekly_tensor, monthly_tensor)
            
        # 计算期望值
        predicted_metrics = {}
        for dim in ['return', 'sharpe', 'drawdown']:
            probs = torch.exp(outputs[dim]).squeeze(0).cpu().numpy()
            if dim == 'return':
                centers = label_generator.return_centers.numpy()
            elif dim == 'sharpe':
                centers = label_generator.sharpe_centers.numpy()
            else:  # drawdown
                centers = label_generator.drawdown_centers.numpy()
            
            predicted_metrics[f'pred_{dim}'] = np.sum(probs * centers)
        
        # 计算真实指标（如果有足够的未来数据）
        future_end_idx = min(index + look_forward_days, len(full_daily_df))
        if future_end_idx > index + 1:
            future_prices = full_daily_df['close'].iloc[index:future_end_idx]
            actual_metrics = label_generator.calculate_future_metrics(future_prices)
            
            if actual_metrics:
                predicted_metrics.update({
                    'actual_return': actual_metrics['total_return'],
                    'actual_sharpe': actual_metrics['sharpe_ratio'],
                    'actual_drawdown': actual_metrics['max_drawdown']
                })
            else:
                predicted_metrics.update({
                    'actual_return': np.nan,
                    'actual_sharpe': np.nan,
                    'actual_drawdown': np.nan
                })
        else:
            predicted_metrics.update({
                'actual_return': np.nan,
                'actual_sharpe': np.nan,
                'actual_drawdown': np.nan
            })
        
        predicted_metrics['date'] = current_date
        results.append(predicted_metrics)

    result_df = pd.DataFrame(results)
    
    # 绘制多子图
    draw_multi_subplot_simple(result_df, model_name, stock_code)
    
    return result_df

def draw_multi_subplot_simple(result_df, model_name, stock_code):
    """简化版多子图显示三个维度"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    dimensions = [
        ('return', '回报率', '%'),
        ('sharpe', '夏普比率', ''),
        ('drawdown', '最大回撤', '%')
    ]
    
    for i, (dim, title, unit) in enumerate(dimensions):
        pred_col = f'pred_{dim}'
        actual_col = f'actual_{dim}'
        
        # 绘制预测值
        axes[i].plot(result_df['date'], result_df[pred_col], 
                    label=f'预测{title}', color='blue', linewidth=2)
        
        # 绘制真实值（去除NaN）
        actual_data = result_df.dropna(subset=[actual_col])
        if not actual_data.empty:
            axes[i].plot(actual_data['date'], actual_data[actual_col], 
                        label=f'真实{title}', color='red', alpha=0.7, linewidth=2)
        
        axes[i].set_title(f'{title} - {model_name}', fontsize=12)
        axes[i].set_ylabel(f'{title} ({unit})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # 格式化y轴
        if unit == '%':
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    axes[-1].set_xlabel('日期')
    plt.suptitle(f'3D模型预测结果 - {stock_code}', fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_simple_{stock_code}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D预测图保存至: {output_path}")
    plt.show()

def draw_combined_simple(result_df, model_name, stock_code):
    """简化版组合时间线图"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 简单的标准化：使用z-score
    def zscore_normalize(data):
        return (data - data.mean()) / data.std()
    
    # 标准化预测数据
    pred_return_norm = zscore_normalize(result_df['pred_return'])
    pred_sharpe_norm = zscore_normalize(result_df['pred_sharpe'])
    pred_drawdown_norm = zscore_normalize(result_df['pred_drawdown'])
    
    # 绘制预测值
    ax.plot(result_df['date'], pred_return_norm, label='预测回报率', color='blue', linewidth=2)
    ax.plot(result_df['date'], pred_sharpe_norm, label='预测夏普比率', color='green', linewidth=2)
    ax.plot(result_df['date'], pred_drawdown_norm, label='预测最大回撤', color='purple', linewidth=2)
    
    # 绘制真实值（如果有）
    actual_data = result_df.dropna(subset=['actual_return', 'actual_sharpe', 'actual_drawdown'])
    if not actual_data.empty:
        actual_return_norm = zscore_normalize(actual_data['actual_return'])
        actual_sharpe_norm = zscore_normalize(actual_data['actual_sharpe'])
        actual_drawdown_norm = zscore_normalize(actual_data['actual_drawdown'])
        
        ax.plot(actual_data['date'], actual_return_norm, label='真实回报率', 
               color='blue', linestyle='--', alpha=0.7)
        ax.plot(actual_data['date'], actual_sharpe_norm, label='真实夏普比率', 
               color='green', linestyle='--', alpha=0.7)
        ax.plot(actual_data['date'], actual_drawdown_norm, label='真实最大回撤', 
               color='purple', linestyle='--', alpha=0.7)
    
    ax.set_title(f'3D模型组合预测 (标准化) - {model_name}', fontsize=14)
    ax.set_xlabel('日期')
    ax.set_ylabel('标准化值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_combined_simple_{stock_code}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"组合时间线图保存至: {output_path}")
    plt.show()

def draw_3d_scatter_simple(result_df, model_name, stock_code):
    """简化版3D散点图（使用matplotlib）"""
    actual_data = result_df.dropna(subset=['actual_return', 'actual_sharpe', 'actual_drawdown'])
    
    if actual_data.empty:
        print("没有足够的真实数据来绘制3D散点图")
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 预测值
    ax.scatter(result_df['pred_return'], result_df['pred_sharpe'], result_df['pred_drawdown'],
              c='blue', alpha=0.6, s=30, label='预测值')
    
    # 真实值
    ax.scatter(actual_data['actual_return'], actual_data['actual_sharpe'], actual_data['actual_drawdown'],
              c='red', alpha=0.8, s=50, label='真实值')
    
    ax.set_xlabel('回报率')
    ax.set_ylabel('夏普比率')
    ax.set_zlabel('最大回撤')
    ax.set_title(f'3D预测散点图 - {model_name} - {stock_code}')
    ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_scatter_simple_{stock_code}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D散点图保存至: {output_path}")
    plt.show()

def find_best_3d_model():
    """查找最佳的3D模型"""
    model_dir = os.path.join(config.MODEL_DIR, "3d_models")
    if not os.path.exists(model_dir):
        return None
    
    # 优先查找总损失最佳模型
    loss_models = glob.glob(os.path.join(model_dir, "best_loss_top*.pth"))
    if loss_models:
        return sorted(loss_models)[0]
    
    # 如果没有，查找任何3D模型
    all_models = glob.glob(os.path.join(model_dir, "*.pth"))
    if all_models:
        return sorted(all_models)[0]
    
    return None

def draw_all_simple_visualizations(stock_code, years=2, model_path=None):
    """绘制所有简化版3D可视化"""
    print(f"开始为股票 {stock_code} 生成简化版3D可视化...")
    
    # 首先获取数据
    result_df = draw_3d_simple(stock_code, years, model_path)
    
    if result_df is None or result_df.empty:
        print("无法获取预测数据")
        return
    
    model_to_load = model_path if model_path else find_best_3d_model()
    model_name = os.path.basename(model_to_load).replace('.pth', '') if model_to_load else "unknown"
    
    # 生成其他可视化
    print("\n正在生成组合时间线图...")
    try:
        draw_combined_simple(result_df, model_name, stock_code)
    except Exception as e:
        print(f"生成组合时间线图时出错: {e}")
    
    print("\n正在生成3D散点图...")
    try:
        draw_3d_scatter_simple(result_df, model_name, stock_code)
    except Exception as e:
        print(f"生成3D散点图时出错: {e}")
    
    print(f"\n所有简化版3D可视化生成完成！")

if __name__ == '__main__':
    # 使用示例
    STOCK_TO_DRAW = config.STOCK_CODES[0] if config.STOCK_CODES else "002415"
    YEARS_TO_DRAW = 5
    
    print("简化版3D可视化选项:")
    print("1. 基础多子图显示")
    print("2. 所有可视化")
    
    # 默认生成基础多子图显示
    draw_3d_simple(STOCK_TO_DRAW, YEARS_TO_DRAW)