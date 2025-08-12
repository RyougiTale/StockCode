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
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import plotly.express as px
from plotly.subplots import make_subplots

# --- 路径设置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

# --- 导入 long_way 模块 ---
from . import config
from .model_3d import create_3d_model
from .data_utils import resample_to_period, calculate_features
from .label_3d_generator import ThreeDimensionalLabelGenerator

def draw_3d_prediction_vs_actual(stock_code, years=2, model_path=None, visualization_type='multi_subplot'):
    """
    为3D模型绘制预测与真实的对比图
    
    Args:
        stock_code: 股票代码
        years: 绘制年数
        model_path: 模型路径
        visualization_type: 可视化类型
            - 'multi_subplot': 多子图显示三个维度
            - 'combined_timeline': 组合时间线图
            - 'scatter_3d': 3D散点图
            - 'radar_chart': 雷达图
            - 'heatmap': 热力图
    """
    
    # 加载模型
    model_to_load = model_path if model_path else find_best_3d_model()
    if not model_to_load:
        print("未找到3D模型文件")
        return
        
    model_name = os.path.basename(model_to_load).replace('.pth', '')
    print(f"加载3D模型: {model_name}")
    
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_to_load, map_location=config.DEVICE))
    model.eval()

    # 数据预处理（与原版保持一致）
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
    
    # 根据可视化类型绘图
    if visualization_type == 'multi_subplot':
        draw_multi_subplot_3d(result_df, model_name, stock_code)
    elif visualization_type == 'combined_timeline':
        draw_combined_timeline_3d(result_df, model_name, stock_code)
    elif visualization_type == 'scatter_3d':
        draw_scatter_3d(result_df, model_name, stock_code)
    elif visualization_type == 'radar_chart':
        draw_radar_chart_3d(result_df, model_name, stock_code)
    elif visualization_type == 'heatmap':
        draw_heatmap_3d(result_df, model_name, stock_code)
    else:
        print(f"未知的可视化类型: {visualization_type}")

def draw_multi_subplot_3d(result_df, model_name, stock_code):
    """方案1: 多子图显示三个维度"""
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
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_multi_{stock_code}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"多子图保存至: {output_path}")
    plt.show()

def draw_combined_timeline_3d(result_df, model_name, stock_code):
    """方案2: 组合时间线图（标准化后显示）"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 标准化数据以便在同一图中显示
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # 准备数据
    pred_data = result_df[['pred_return', 'pred_sharpe', 'pred_drawdown']].values
    pred_scaled = scaler.fit_transform(pred_data)
    
    # 绘制预测值
    ax.plot(result_df['date'], pred_scaled[:, 0], label='预测回报率', color='blue', linewidth=2)
    ax.plot(result_df['date'], pred_scaled[:, 1], label='预测夏普比率', color='green', linewidth=2)
    ax.plot(result_df['date'], pred_scaled[:, 2], label='预测最大回撤', color='purple', linewidth=2)
    
    # 绘制真实值（如果有）
    actual_data = result_df.dropna(subset=['actual_return', 'actual_sharpe', 'actual_drawdown'])
    if not actual_data.empty:
        actual_values = actual_data[['actual_return', 'actual_sharpe', 'actual_drawdown']].values
        actual_scaled = scaler.transform(actual_values)
        
        ax.plot(actual_data['date'], actual_scaled[:, 0], label='真实回报率', 
               color='blue', linestyle='--', alpha=0.7)
        ax.plot(actual_data['date'], actual_scaled[:, 1], label='真实夏普比率', 
               color='green', linestyle='--', alpha=0.7)
        ax.plot(actual_data['date'], actual_scaled[:, 2], label='真实最大回撤', 
               color='purple', linestyle='--', alpha=0.7)
    
    ax.set_title(f'3D模型组合预测 (标准化) - {model_name}', fontsize=14)
    ax.set_xlabel('日期')
    ax.set_ylabel('标准化值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_combined_{stock_code}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"组合时间线图保存至: {output_path}")
    plt.show()

def draw_scatter_3d(result_df, model_name, stock_code):
    """方案3: 3D散点图"""
    # 使用plotly创建交互式3D散点图
    actual_data = result_df.dropna(subset=['actual_return', 'actual_sharpe', 'actual_drawdown'])
    
    if actual_data.empty:
        print("没有足够的真实数据来绘制3D散点图")
        return
    
    fig = go.Figure()
    
    # 预测值
    fig.add_trace(go.Scatter3d(
        x=result_df['pred_return'],
        y=result_df['pred_sharpe'],
        z=result_df['pred_drawdown'],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.6),
        name='预测值',
        text=result_df['date'].dt.strftime('%Y-%m-%d'),
        hovertemplate='<b>预测值</b><br>' +
                      '日期: %{text}<br>' +
                      '回报率: %{x:.2%}<br>' +
                      '夏普比率: %{y:.3f}<br>' +
                      '最大回撤: %{z:.2%}<extra></extra>'
    ))
    
    # 真实值
    fig.add_trace(go.Scatter3d(
        x=actual_data['actual_return'],
        y=actual_data['actual_sharpe'],
        z=actual_data['actual_drawdown'],
        mode='markers',
        marker=dict(size=6, color='red', opacity=0.8),
        name='真实值',
        text=actual_data['date'].dt.strftime('%Y-%m-%d'),
        hovertemplate='<b>真实值</b><br>' +
                      '日期: %{text}<br>' +
                      '回报率: %{x:.2%}<br>' +
                      '夏普比率: %{y:.3f}<br>' +
                      '最大回撤: %{z:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'3D预测散点图 - {model_name} - {stock_code}',
        scene=dict(
            xaxis_title='回报率',
            yaxis_title='夏普比率',
            zaxis_title='最大回撤',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )
    
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_scatter_{stock_code}.html")
    fig.write_html(output_path)
    print(f"3D散点图保存至: {output_path}")
    fig.show()

def draw_radar_chart_3d(result_df, model_name, stock_code):
    """方案4: 雷达图显示最近的预测结果"""
    # 取最近的10个预测结果
    recent_data = result_df.tail(10)
    
    fig = go.Figure()
    
    # 为每个时间点创建雷达图
    for i, (_, row) in enumerate(recent_data.iterrows()):
        # 标准化数据到0-1范围以便显示
        values = [
            (row['pred_return'] + 0.1) / 0.2,  # 回报率 [-0.1, 0.1] -> [0, 1]
            (row['pred_sharpe'] + 1.0) / 2.0,  # 夏普比率 [-1, 1] -> [0, 1]
            (row['pred_drawdown'] + 0.2) / 0.2  # 最大回撤 [-0.2, 0] -> [0, 1]
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 闭合图形
            theta=['回报率', '夏普比率', '最大回撤', '回报率'],
            fill='toself',
            name=f"{row['date'].strftime('%m-%d')}",
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title=f'最近预测雷达图 - {model_name} - {stock_code}',
        showlegend=True
    )
    
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_radar_{stock_code}.html")
    fig.write_html(output_path)
    print(f"雷达图保存至: {output_path}")
    fig.show()

def draw_heatmap_3d(result_df, model_name, stock_code):
    """方案5: 热力图显示预测准确性"""
    actual_data = result_df.dropna(subset=['actual_return', 'actual_sharpe', 'actual_drawdown'])
    
    if actual_data.empty:
        print("没有足够的真实数据来绘制热力图")
        return
    
    # 计算预测误差
    errors = pd.DataFrame({
        '回报率误差': np.abs(actual_data['pred_return'] - actual_data['actual_return']),
        '夏普比率误差': np.abs(actual_data['pred_sharpe'] - actual_data['actual_sharpe']),
        '最大回撤误差': np.abs(actual_data['pred_drawdown'] - actual_data['actual_drawdown']),
        '日期': actual_data['date']
    })
    
    # 按月份聚合
    errors['月份'] = errors['日期'].dt.to_period('M')
    monthly_errors = errors.groupby('月份')[['回报率误差', '夏普比率误差', '最大回撤误差']].mean()
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(monthly_errors.T, annot=True, fmt='.4f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': '平均绝对误差'})
    
    ax.set_title(f'3D预测误差热力图 - {model_name} - {stock_code}', fontsize=14)
    ax.set_xlabel('月份')
    ax.set_ylabel('指标')
    
    plt.tight_layout()
    output_path = os.path.join(config.MODEL_DIR, f"3d_prediction_heatmap_{stock_code}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图保存至: {output_path}")
    plt.show()

def find_best_3d_model():
    """查找最佳的3D模型"""
    model_dir = os.path.join(config.MODEL_DIR, "3d_models")
    if not os.path.exists(model_dir):
        return None
    
    # 优先查找总损失最佳模型
    loss_models = glob.glob(os.path.join(model_dir, "best_total_loss_*.pth"))
    if loss_models:
        return sorted(loss_models)[0]  # 返回第一个
    
    # 如果没有，查找任何3D模型
    all_models = glob.glob(os.path.join(model_dir, "*.pth"))
    if all_models:
        return sorted(all_models)[0]
    
    return None

def draw_all_3d_visualizations(stock_code, years=2, model_path=None):
    """绘制所有类型的3D可视化"""
    print(f"开始为股票 {stock_code} 生成所有3D可视化...")
    
    visualizations = [
        ('multi_subplot', '多子图显示'),
        ('combined_timeline', '组合时间线'),
        ('scatter_3d', '3D散点图'),
        ('radar_chart', '雷达图'),
        ('heatmap', '热力图')
    ]
    
    for viz_type, viz_name in visualizations:
        print(f"\n正在生成 {viz_name}...")
        try:
            draw_3d_prediction_vs_actual(stock_code, years, model_path, viz_type)
        except Exception as e:
            print(f"生成 {viz_name} 时出错: {e}")
    
    print(f"\n所有3D可视化生成完成！")

if __name__ == '__main__':
    # 使用示例
    STOCK_TO_DRAW = config.STOCK_CODES[0] if config.STOCK_CODES else "002415"
    YEARS_TO_DRAW = 2
    
    print("3D可视化选项:")
    print("1. 多子图显示 (multi_subplot)")
    print("2. 组合时间线 (combined_timeline)")
    print("3. 3D散点图 (scatter_3d)")
    print("4. 雷达图 (radar_chart)")
    print("5. 热力图 (heatmap)")
    print("6. 所有可视化 (all)")
    
    # 默认生成多子图显示
    draw_3d_prediction_vs_actual(STOCK_TO_DRAW, YEARS_TO_DRAW, visualization_type='multi_subplot')