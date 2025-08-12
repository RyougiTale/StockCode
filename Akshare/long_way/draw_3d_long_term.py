#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D模型长期预测vs实际对比可视化工具
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入父目录的stock_util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

# 导入long_way模块（使用相对导入，需要用 python -m long_way.draw_3d_long_term 运行）
from . import config
from .model_3d import create_3d_model
from .data_utils import resample_to_period, calculate_features, get_all_samples
from .rolling_scaler import RollingWindowScaler
from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from .logger_config import get_logger

# 设置中文字体和编码
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# matplotlib.rcParams['font.size'] = 10

# 设置图表风格
# plt.style.use('default')  # 使用默认风格，避免seaborn兼容性问题
# plt.rcParams['figure.facecolor'] = 'white'

logger = get_logger(__name__)

def reverse_relative_mapping(relative_pos, centers):
    """
    反向分段线性插值：从相对位置[0,1]映射回绝对值
    这是improved_label_generator.py中convert_to_relative的逆过程
    
    原始映射算法：
    - relative_pos = (i + progress) / (len(centers) - 1)
    - progress = (value - centers[i]) / (centers[i+1] - centers[i])
    
    反向算法：
    - i + progress = relative_pos * (len(centers) - 1)
    - i = floor(relative_pos * (len(centers) - 1))
    - progress = relative_pos * (len(centers) - 1) - i
    - value = centers[i] + progress * (centers[i+1] - centers[i])
    """
    centers = np.array(centers)
    relative_pos = np.clip(relative_pos, 0.0, 1.0)
    
    # 边界处理
    if relative_pos <= 0.0:
        return centers[0]
    elif relative_pos >= 1.0:
        return centers[-1]
    
    # 计算在哪个区间
    scaled_pos = relative_pos * (len(centers) - 1)
    i = int(np.floor(scaled_pos))
    
    # 确保不超出数组边界
    if i >= len(centers) - 1:
        return centers[-1]
    
    # 计算区间内的进度
    progress = scaled_pos - i
    
    # 线性插值
    value = centers[i] + progress * (centers[i + 1] - centers[i])
    return value

def calculate_actual_3d_metrics(df, look_forward_days=20):
    """
    计算实际的3D指标（收益率、夏普比率、最大回撤）
    """
    df = df.copy()
    results = []
    
    for i in range(len(df) - look_forward_days):
        current_date = df.iloc[i]['date']
        current_price = df.iloc[i]['close']
        
        # 获取未来20天的价格序列
        future_prices = df.iloc[i:i+look_forward_days+1]['close'].values
        
        if len(future_prices) < look_forward_days + 1:
            continue
            
        # 计算实际指标 - 与训练时的标签生成器保持完全一致
        price_series = pd.Series(future_prices)
        
        # 1) 总回报率 - 与训练时一致
        total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1.0
        
        # 2) 夏普比率 - 与训练时一致（年化）
        daily_returns = price_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # 年化！
        else:
            sharpe_ratio = 0.0
            
        # 3) 最大回撤 - 与训练时一致
        cumulative_max = price_series.cummax()
        drawdown = (price_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        results.append({
            'date': current_date,
            'actual_return': total_return,
            'actual_sharpe': sharpe_ratio,
            'actual_drawdown': max_drawdown
        })
    
    return pd.DataFrame(results)

def predict_3d_long_term(stock_code, model_path, years=3):
    """
    生成3D模型的长期预测vs实际对比数据
    """
    logger.info(f"正在为 {stock_code} 生成长期3D预测对比...")
    
    # 加载模型
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # 重建训练时的股票分布，以获得正确的自适应中心点
    logger.info("重建股票历史分布以获取正确的自适应中心点...")
    
    # 创建标签生成器并构建股票分布
    # 获取所有样本用于构建分布
    logger.info("获取历史样本数据以构建股票分布...")
    all_samples, _ = get_all_samples([stock_code])  # 只获取当前股票的样本
    
    if not all_samples:
        logger.warning("无法获取样本数据，将使用基准中心点")
        # 回退到基准中心点
        return_centers = np.array([-0.15, -0.05, 0.02, 0.08, 0.20])
        sharpe_centers = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        drawdown_centers = np.array([-0.25, -0.15, -0.08, -0.04, -0.01])
    else:
        # 创建标签生成器并构建分布
        label_generator = ImprovedThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        
        # 按股票分组样本
        stock_samples_dict = {}
        for sample in all_samples:
            stock_code_sample = sample.get('stock_code', stock_code)
            if stock_code_sample not in stock_samples_dict:
                stock_samples_dict[stock_code_sample] = []
            stock_samples_dict[stock_code_sample].append(sample)
        
        # 构建股票分布
        label_generator.fit_stock_distributions(stock_samples_dict)
        logger.info(f"成功构建了 {len(label_generator.relative_calculator.stock_distributions)} 只股票的分布")
    
    # 数据预处理
    daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
    monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)
    
    full_daily_df = read_history_by_code(stock_code)
    if full_daily_df is None or full_daily_df.empty:
        raise ValueError(f"无法获取股票 {stock_code} 的数据")
    
    # 计算实际3D指标
    logger.info("计算实际3D指标...")
    actual_metrics_df = calculate_actual_3d_metrics(
        full_daily_df, 
        config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    )
    
    # 特征工程
    daily_featured = calculate_features(full_daily_df.copy(), 'daily')
    weekly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'W-FRI'), 'weekly')
    monthly_featured = calculate_features(resample_to_period(full_daily_df.copy(), 'ME'), 'monthly')
    
    # 归一化
    daily_featured = daily_scaler.fit_transform(daily_featured, config.FEATURE_COLUMNS['daily'])
    weekly_featured = weekly_scaler.fit_transform(weekly_featured, config.FEATURE_COLUMNS['weekly'])
    monthly_featured = monthly_scaler.fit_transform(monthly_featured, config.FEATURE_COLUMNS['monthly'])
    
    # 确定时间范围
    data_end_date = full_daily_df['date'].max()
    start_date = data_end_date - pd.DateOffset(years=years)
    
    # 历史数据用于有实际对比的预测
    historical_df = full_daily_df[full_daily_df['date'] >= start_date].copy()
    
    # 添加未来预测点：基于最新数据预测未来20天
    look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    
    # 创建未来日期序列（工作日）
    future_dates = []
    current_date = data_end_date
    days_added = 0
    while days_added < look_forward_days:
        current_date += pd.Timedelta(days=1)
        # 简单假设工作日（实际可以更精确地排除节假日）
        if current_date.weekday() < 5:  # 0-4 是周一到周五
            future_dates.append(current_date)
            days_added += 1
    
    # 合并历史数据和未来预测日期
    target_df = historical_df.copy()
    
    # 生成预测
    predictions = []
    logger.info(f"生成预测数据:")
    logger.info(f"  历史对比期间: {start_date.date()} 到 {data_end_date.date()}")
    logger.info(f"  未来预测期间: {future_dates[0].date()} 到 {future_dates[-1].date()}")
    
    for index, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"预测 {stock_code}"):
        current_date = row['date']
        
        # 获取输入数据切片
        daily_slice = daily_featured[daily_featured['date'] <= current_date].tail(config.DAILY_SEQ_LEN)
        weekly_slice = weekly_featured[weekly_featured['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        monthly_slice = monthly_featured[monthly_featured['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        
        # 检查数据长度
        if not (len(daily_slice) == config.DAILY_SEQ_LEN and 
                len(weekly_slice) == config.WEEKLY_SEQ_LEN and 
                len(monthly_slice) == config.MONTHLY_SEQ_LEN):
            continue
        
        # 转换为tensor
        daily_tensor = torch.from_numpy(
            daily_slice[config.FEATURE_COLUMNS['daily']].values.astype(np.float32)
        ).unsqueeze(0).to(config.DEVICE)
        
        weekly_tensor = torch.from_numpy(
            weekly_slice[config.FEATURE_COLUMNS['weekly']].values.astype(np.float32)
        ).unsqueeze(0).to(config.DEVICE)
        
        monthly_tensor = torch.from_numpy(
            monthly_slice[config.FEATURE_COLUMNS['monthly']].values.astype(np.float32)
        ).unsqueeze(0).to(config.DEVICE)
        
        # 模型预测
        with torch.no_grad():
            output = model(daily_tensor, weekly_tensor, monthly_tensor)
            
            # 提取3D预测结果（模型输出的已经是log_softmax，需要exp得到概率）
            return_probs = torch.exp(output['return']).cpu().numpy()[0]
            sharpe_probs = torch.exp(output['sharpe']).cpu().numpy()[0] 
            drawdown_probs = torch.exp(output['drawdown']).cpu().numpy()[0]
            
            # 使用与训练时一致的相对中心点（修复训练/预测空间不一致问题）
            # 训练时使用的是固定的相对中心点 [0.0, 0.25, 0.5, 0.75, 1.0]
            relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            
            # 计算相对空间的预期值
            expected_return_relative = np.sum(return_probs * relative_centers)
            expected_sharpe_relative = np.sum(sharpe_probs * relative_centers)
            expected_drawdown_relative = np.sum(drawdown_probs * relative_centers)
            
            # 获取自适应中心点用于反向映射到绝对值
            if 'label_generator' in locals():
                return_centers = np.array(label_generator.relative_calculator.get_adaptive_centers(stock_code, 'total_return'))
                sharpe_centers = np.array(label_generator.relative_calculator.get_adaptive_centers(stock_code, 'sharpe_ratio'))  
                drawdown_centers = np.array(label_generator.relative_calculator.get_adaptive_centers(stock_code, 'max_drawdown'))
                
                # 通过反向分段线性插值将相对值映射回绝对值
                expected_return = reverse_relative_mapping(expected_return_relative, return_centers)
                expected_sharpe = reverse_relative_mapping(expected_sharpe_relative, sharpe_centers)
                expected_drawdown = reverse_relative_mapping(expected_drawdown_relative, drawdown_centers)
            else:
                # 回退到基准中心点进行映射
                return_centers_baseline = np.array([-0.15, -0.05, 0.02, 0.08, 0.20])
                sharpe_centers_baseline = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
                drawdown_centers_baseline = np.array([-0.25, -0.15, -0.08, -0.04, -0.01])
                
                expected_return = reverse_relative_mapping(expected_return_relative, return_centers_baseline)
                expected_sharpe = reverse_relative_mapping(expected_sharpe_relative, sharpe_centers_baseline)
                expected_drawdown = reverse_relative_mapping(expected_drawdown_relative, drawdown_centers_baseline)
            
            # 计算不同Top-K预测（也使用相对中心点然后映射）
            # Top-1预测
            return_top1_relative = relative_centers[np.argmax(return_probs)]
            sharpe_top1_relative = relative_centers[np.argmax(sharpe_probs)]
            drawdown_top1_relative = relative_centers[np.argmax(drawdown_probs)]
            
            # Top-2加权预测
            return_top2_idx = np.argsort(return_probs)[-2:]
            return_top2_probs = return_probs[return_top2_idx]
            return_top2_probs_norm = return_top2_probs / np.sum(return_top2_probs)
            return_top2_relative = np.sum(return_top2_probs_norm * relative_centers[return_top2_idx])
            
            # Top-3加权预测
            return_top3_idx = np.argsort(return_probs)[-3:]
            return_top3_probs = return_probs[return_top3_idx]
            return_top3_probs_norm = return_top3_probs / np.sum(return_top3_probs)
            return_top3_relative = np.sum(return_top3_probs_norm * relative_centers[return_top3_idx])
            
            sharpe_top2_idx = np.argsort(sharpe_probs)[-2:]
            sharpe_top2_probs = sharpe_probs[sharpe_top2_idx]
            sharpe_top2_probs_norm = sharpe_top2_probs / np.sum(sharpe_top2_probs)
            sharpe_top2_relative = np.sum(sharpe_top2_probs_norm * relative_centers[sharpe_top2_idx])
            
            sharpe_top3_idx = np.argsort(sharpe_probs)[-3:]
            sharpe_top3_probs = sharpe_probs[sharpe_top3_idx]
            sharpe_top3_probs_norm = sharpe_top3_probs / np.sum(sharpe_top3_probs)
            sharpe_top3_relative = np.sum(sharpe_top3_probs_norm * relative_centers[sharpe_top3_idx])
            
            drawdown_top2_idx = np.argsort(drawdown_probs)[-2:]
            drawdown_top2_probs = drawdown_probs[drawdown_top2_idx]
            drawdown_top2_probs_norm = drawdown_top2_probs / np.sum(drawdown_top2_probs)
            drawdown_top2_relative = np.sum(drawdown_top2_probs_norm * relative_centers[drawdown_top2_idx])
            
            drawdown_top3_idx = np.argsort(drawdown_probs)[-3:]
            drawdown_top3_probs = drawdown_probs[drawdown_top3_idx]
            drawdown_top3_probs_norm = drawdown_top3_probs / np.sum(drawdown_top3_probs)
            drawdown_top3_relative = np.sum(drawdown_top3_probs_norm * relative_centers[drawdown_top3_idx])
            
            # 将所有相对值映射到绝对值
            if 'label_generator' in locals():
                return_top1 = reverse_relative_mapping(return_top1_relative, return_centers)
                return_top2 = reverse_relative_mapping(return_top2_relative, return_centers)
                return_top3 = reverse_relative_mapping(return_top3_relative, return_centers)
                
                sharpe_top1 = reverse_relative_mapping(sharpe_top1_relative, sharpe_centers)
                sharpe_top2 = reverse_relative_mapping(sharpe_top2_relative, sharpe_centers)
                sharpe_top3 = reverse_relative_mapping(sharpe_top3_relative, sharpe_centers)
                
                drawdown_top1 = reverse_relative_mapping(drawdown_top1_relative, drawdown_centers)
                drawdown_top2 = reverse_relative_mapping(drawdown_top2_relative, drawdown_centers)
                drawdown_top3 = reverse_relative_mapping(drawdown_top3_relative, drawdown_centers)
            else:
                # 回退到基准中心点进行映射
                return_centers_baseline = np.array([-0.15, -0.05, 0.02, 0.08, 0.20])
                sharpe_centers_baseline = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
                drawdown_centers_baseline = np.array([-0.25, -0.15, -0.08, -0.04, -0.01])
                
                return_top1 = reverse_relative_mapping(return_top1_relative, return_centers_baseline)
                return_top2 = reverse_relative_mapping(return_top2_relative, return_centers_baseline)
                return_top3 = reverse_relative_mapping(return_top3_relative, return_centers_baseline)
                
                sharpe_top1 = reverse_relative_mapping(sharpe_top1_relative, sharpe_centers_baseline)
                sharpe_top2 = reverse_relative_mapping(sharpe_top2_relative, sharpe_centers_baseline)
                sharpe_top3 = reverse_relative_mapping(sharpe_top3_relative, sharpe_centers_baseline)
                
                drawdown_top1 = reverse_relative_mapping(drawdown_top1_relative, drawdown_centers_baseline)
                drawdown_top2 = reverse_relative_mapping(drawdown_top2_relative, drawdown_centers_baseline)
                drawdown_top3 = reverse_relative_mapping(drawdown_top3_relative, drawdown_centers_baseline)
            
            predictions.append({
                'date': current_date,
                'pred_return_full': expected_return,
                'pred_return_top3': return_top3,
                'pred_return_top2': return_top2,
                'pred_return_top1': return_top1,
                'pred_sharpe_full': expected_sharpe,
                'pred_sharpe_top3': sharpe_top3,
                'pred_sharpe_top2': sharpe_top2,
                'pred_sharpe_top1': sharpe_top1,
                'pred_drawdown_full': expected_drawdown,
                'pred_drawdown_top3': drawdown_top3,
                'pred_drawdown_top2': drawdown_top2,
                'pred_drawdown_top1': drawdown_top1
            })
    
    # 添加未来预测（基于最新数据）
    logger.info("生成未来20天预测...")
    if len(target_df) > 0:
        # 使用最后一个完整的数据点进行未来预测
        last_date = data_end_date
        last_daily_slice = daily_featured[daily_featured['date'] <= last_date].tail(config.DAILY_SEQ_LEN)
        last_weekly_slice = weekly_featured[weekly_featured['date'] <= last_date].tail(config.WEEKLY_SEQ_LEN)
        last_monthly_slice = monthly_featured[monthly_featured['date'] <= last_date].tail(config.MONTHLY_SEQ_LEN)
        
        # 检查数据完整性
        if (len(last_daily_slice) == config.DAILY_SEQ_LEN and 
            len(last_weekly_slice) == config.WEEKLY_SEQ_LEN and 
            len(last_monthly_slice) == config.MONTHLY_SEQ_LEN):
            
            # 转换为tensor
            daily_tensor = torch.from_numpy(
                last_daily_slice[config.FEATURE_COLUMNS['daily']].values.astype(np.float32)
            ).unsqueeze(0).to(config.DEVICE)
            
            weekly_tensor = torch.from_numpy(
                last_weekly_slice[config.FEATURE_COLUMNS['weekly']].values.astype(np.float32)
            ).unsqueeze(0).to(config.DEVICE)
            
            monthly_tensor = torch.from_numpy(
                last_monthly_slice[config.FEATURE_COLUMNS['monthly']].values.astype(np.float32)
            ).unsqueeze(0).to(config.DEVICE)
            
            # 模型预测
            with torch.no_grad():
                output = model(daily_tensor, weekly_tensor, monthly_tensor)
                
                # 模型输出的已经是log_softmax，需要exp得到概率
                return_probs = torch.exp(output['return']).cpu().numpy()[0]
                sharpe_probs = torch.exp(output['sharpe']).cpu().numpy()[0] 
                drawdown_probs = torch.exp(output['drawdown']).cpu().numpy()[0]
                
                # 获取中心点
                if 'label_generator' in locals():
                    return_centers = np.array(label_generator.relative_calculator.get_adaptive_centers(stock_code, 'total_return'))
                    sharpe_centers = np.array(label_generator.relative_calculator.get_adaptive_centers(stock_code, 'sharpe_ratio'))  
                    drawdown_centers = np.array(label_generator.relative_calculator.get_adaptive_centers(stock_code, 'max_drawdown'))
                else:
                    return_centers = np.array([-0.15, -0.05, 0.02, 0.08, 0.20])
                    sharpe_centers = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
                    drawdown_centers = np.array([-0.25, -0.15, -0.08, -0.04, -0.01])
                
                # 使用相对中心点然后映射到绝对值（与历史预测保持一致）
                relative_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                
                # 计算相对空间的预期值
                expected_return_relative = np.sum(return_probs * relative_centers)
                expected_sharpe_relative = np.sum(sharpe_probs * relative_centers)
                expected_drawdown_relative = np.sum(drawdown_probs * relative_centers)
                
                # 映射到绝对值
                expected_return = reverse_relative_mapping(expected_return_relative, return_centers)
                expected_sharpe = reverse_relative_mapping(expected_sharpe_relative, sharpe_centers)
                expected_drawdown = reverse_relative_mapping(expected_drawdown_relative, drawdown_centers)
                
                # Top-1预测
                return_top1_relative = relative_centers[np.argmax(return_probs)]
                sharpe_top1_relative = relative_centers[np.argmax(sharpe_probs)]
                drawdown_top1_relative = relative_centers[np.argmax(drawdown_probs)]
                
                return_top1 = reverse_relative_mapping(return_top1_relative, return_centers)
                sharpe_top1 = reverse_relative_mapping(sharpe_top1_relative, sharpe_centers)
                drawdown_top1 = reverse_relative_mapping(drawdown_top1_relative, drawdown_centers)
                
                # Top-2计算
                return_top2_idx = np.argsort(return_probs)[-2:]
                return_top2_probs_norm = return_probs[return_top2_idx] / np.sum(return_probs[return_top2_idx])
                return_top2_relative = np.sum(return_top2_probs_norm * relative_centers[return_top2_idx])
                return_top2 = reverse_relative_mapping(return_top2_relative, return_centers)
                
                sharpe_top2_idx = np.argsort(sharpe_probs)[-2:]
                sharpe_top2_probs_norm = sharpe_probs[sharpe_top2_idx] / np.sum(sharpe_probs[sharpe_top2_idx])
                sharpe_top2_relative = np.sum(sharpe_top2_probs_norm * relative_centers[sharpe_top2_idx])
                sharpe_top2 = reverse_relative_mapping(sharpe_top2_relative, sharpe_centers)
                
                drawdown_top2_idx = np.argsort(drawdown_probs)[-2:]
                drawdown_top2_probs_norm = drawdown_probs[drawdown_top2_idx] / np.sum(drawdown_probs[drawdown_top2_idx])
                drawdown_top2_relative = np.sum(drawdown_top2_probs_norm * relative_centers[drawdown_top2_idx])
                drawdown_top2 = reverse_relative_mapping(drawdown_top2_relative, drawdown_centers)
                
                # Top-3计算
                return_top3_idx = np.argsort(return_probs)[-3:]
                return_top3_probs_norm = return_probs[return_top3_idx] / np.sum(return_probs[return_top3_idx])
                return_top3_relative = np.sum(return_top3_probs_norm * relative_centers[return_top3_idx])
                return_top3 = reverse_relative_mapping(return_top3_relative, return_centers)
                
                sharpe_top3_idx = np.argsort(sharpe_probs)[-3:]
                sharpe_top3_probs_norm = sharpe_probs[sharpe_top3_idx] / np.sum(sharpe_probs[sharpe_top3_idx])
                sharpe_top3_relative = np.sum(sharpe_top3_probs_norm * relative_centers[sharpe_top3_idx])
                sharpe_top3 = reverse_relative_mapping(sharpe_top3_relative, sharpe_centers)
                
                drawdown_top3_idx = np.argsort(drawdown_probs)[-3:]
                drawdown_top3_probs_norm = drawdown_probs[drawdown_top3_idx] / np.sum(drawdown_probs[drawdown_top3_idx])
                drawdown_top3_relative = np.sum(drawdown_top3_probs_norm * relative_centers[drawdown_top3_idx])
                drawdown_top3 = reverse_relative_mapping(drawdown_top3_relative, drawdown_centers)
                
                # 为每个未来日期添加相同的预测（这是未来20天整体的预测）
                for future_date in future_dates:
                    predictions.append({
                        'date': future_date,
                        'pred_return_full': expected_return,
                        'pred_return_top3': return_top3,
                        'pred_return_top2': return_top2,
                        'pred_return_top1': return_top1,
                        'pred_sharpe_full': expected_sharpe,
                        'pred_sharpe_top3': sharpe_top3,
                        'pred_sharpe_top2': sharpe_top2,
                        'pred_sharpe_top1': sharpe_top1,
                        'pred_drawdown_full': expected_drawdown,
                        'pred_drawdown_top3': drawdown_top3,
                        'pred_drawdown_top2': drawdown_top2,
                        'pred_drawdown_top1': drawdown_top1,
                        'is_future': True  # 标记为未来预测
                    })
    
    pred_df = pd.DataFrame(predictions)
    
    # 合并预测和实际数据（只有历史部分有实际数据）
    merged_df = pd.merge(pred_df, actual_metrics_df, on='date', how='left')
    
    logger.info(f"生成了 {len(merged_df)} 个有效的预测vs实际对比点")
    return merged_df

def draw_3d_long_term_comparison(stock_code, model_path=None, years=3, save_path=None):
    """
    绘制3D模型的长期预测vs实际对比图
    
    Args:
        stock_code: 股票代码
        model_path: 模型文件路径，如果为None则使用默认的best_loss_top_1.pth
        years: 显示的年数
        save_path: 保存图片的路径
    """
    # 确定模型路径 - 使用统一的标准路径
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "best_loss_top_1.pth")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model_name = os.path.basename(model_path).replace('.pth', '')
    logger.info(f"使用模型: {model_name}")
    
    # 生成预测数据
    df = predict_3d_long_term(stock_code, model_path, years)
    
    if df.empty:
        logger.error("没有生成有效的预测数据")
        return
    
    # 分离历史数据和未来预测数据
    df_historical = df[df['actual_return'].notna()].copy()  # 有实际数据的历史部分
    df_future = df[df['actual_return'].isna()].copy()      # 未来预测部分
    
    # 获取市场周期配置用于标记训练/验证/测试集
    from datetime import datetime, timedelta
    current_time = datetime.now()
    recent_cutoff = current_time - timedelta(days=365 * config.MARKET_PERIOD_CONFIG["recent_years"])
    middle_cutoff = current_time - timedelta(days=365 * config.MARKET_PERIOD_CONFIG["middle_years"])
    
    # 计算每个时期内的划分（80%训练，10%验证，10%测试）
    train_ratio = 0.8
    val_ratio = 0.1
    
    # 为历史数据添加数据集标记
    def get_dataset_type(date):
        """根据日期和时期内位置确定数据集类型"""
        # 先确定属于哪个时期
        if date >= recent_cutoff:
            period = 'recent'
            period_start = recent_cutoff
            period_end = current_time
        elif date >= middle_cutoff:
            period = 'middle'
            period_start = middle_cutoff  
            period_end = recent_cutoff
        else:
            period = 'distant'
            # 对于distant时期，使用数据的最早时间作为起点
            period_start = df_historical['date'].min()
            period_end = middle_cutoff
        
        # 计算在时期内的相对位置
        total_days = (period_end - period_start).days
        if total_days <= 0:
            return 'test'  # 默认
        
        position_days = (date - period_start).days
        relative_position = position_days / total_days
        
        # 根据相对位置判断属于训练/验证/测试
        if relative_position < train_ratio:
            return 'train'
        elif relative_position < train_ratio + val_ratio:
            return 'val'
        else:
            return 'test'
    
    # 应用标记
    if not df_historical.empty:
        df_historical['dataset_type'] = df_historical['date'].apply(get_dataset_type)
    
    logger.info(f"历史对比数据: {len(df_historical)} 个点")
    if not df_historical.empty and 'dataset_type' in df_historical.columns:
        train_count = len(df_historical[df_historical['dataset_type'] == 'train'])
        val_count = len(df_historical[df_historical['dataset_type'] == 'val'])
        test_count = len(df_historical[df_historical['dataset_type'] == 'test'])
        logger.info(f"  - 训练集: {train_count} 个点")
        logger.info(f"  - 验证集: {val_count} 个点")
        logger.info(f"  - 测试集: {test_count} 个点")
    logger.info(f"未来预测数据: {len(df_future)} 个点")
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. 收益率对比
    ax1 = axes[0]
    # 历史预测线（实线）
    if not df_historical.empty:
        ax1.plot(df_historical['date'], df_historical['pred_return_full'] * 100, 
                 label='历史预测 (全概率)', color='royalblue', alpha=0.7, linewidth=2)
        ax1.plot(df_historical['date'], df_historical['pred_return_top3'] * 100, 
                 label='历史预测 (Top-3)', color='darkviolet', linewidth=1.5, linestyle='--')
        ax1.plot(df_historical['date'], df_historical['pred_return_top2'] * 100, 
                 label='历史预测 (Top-2)', color='purple', linewidth=1.5, linestyle='-.')
        ax1.plot(df_historical['date'], df_historical['pred_return_top1'] * 100, 
                 label='历史预测 (Top-1)', color='red', linewidth=1.5, linestyle=':')
        ax1.plot(df_historical['date'], df_historical['actual_return'] * 100, 
                 label='实际表现', color='darkorange', alpha=0.8, linewidth=2.5)
    
    # 未来预测线（虚线，不同颜色）
    if not df_future.empty:
        ax1.plot(df_future['date'], df_future['pred_return_full'] * 100, 
                 label='未来预测 (全概率)', color='lightblue', alpha=0.8, linewidth=2, linestyle='-')
        ax1.plot(df_future['date'], df_future['pred_return_top3'] * 100, 
                 label='未来预测 (Top-3)', color='plum', linewidth=1.5, linestyle='--')
        ax1.plot(df_future['date'], df_future['pred_return_top2'] * 100, 
                 label='未来预测 (Top-2)', color='mediumpurple', linewidth=1.5, linestyle='-.')
        ax1.plot(df_future['date'], df_future['pred_return_top1'] * 100, 
                 label='未来预测 (Top-1)', color='lightcoral', linewidth=1.5, linestyle=':')
    
    # 添加分界线
    if not df_historical.empty and not df_future.empty:
        boundary_date = df_historical['date'].max()
        ax1.axvline(x=boundary_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='预测边界')
    
    # 添加训练/验证/测试集的背景色块
    if not df_historical.empty and 'dataset_type' in df_historical.columns:
        # 为每个数据集类型的连续区间添加背景
        for i in range(len(df_historical)):
            if i == 0 or df_historical.iloc[i]['dataset_type'] != df_historical.iloc[i-1]['dataset_type']:
                # 找到连续相同类型的区间
                start_date = df_historical.iloc[i]['date']
                dataset_type = df_historical.iloc[i]['dataset_type']
                
                # 找到结束位置
                end_idx = i
                while end_idx < len(df_historical) - 1 and df_historical.iloc[end_idx + 1]['dataset_type'] == dataset_type:
                    end_idx += 1
                end_date = df_historical.iloc[end_idx]['date']
                
                # 根据类型设置颜色
                if dataset_type == 'train':
                    ax1.axvspan(start_date, end_date, alpha=0.1, color='blue')
                elif dataset_type == 'val':
                    ax1.axvspan(start_date, end_date, alpha=0.1, color='green')
                elif dataset_type == 'test':
                    ax1.axvspan(start_date, end_date, alpha=0.1, color='orange')
        
        # 添加图例（只添加一次）
        from matplotlib.patches import Patch
        train_patch = Patch(color='blue', alpha=0.1, label='训练集')
        val_patch = Patch(color='green', alpha=0.1, label='验证集')
        test_patch = Patch(color='orange', alpha=0.1, label='测试集')
        ax1.legend(handles=ax1.get_legend_handles_labels()[0] + [train_patch, val_patch, test_patch], 
                   loc='upper left', fontsize=10)
    
    ax1.set_title(f'{stock_code} - 未来20天收益率预测 vs 实际', fontsize=14, fontweight='bold')
    ax1.set_ylabel('收益率 (%)', fontsize=12)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 夏普比率对比
    ax2 = axes[1]
    # 历史预测线
    if not df_historical.empty:
        ax2.plot(df_historical['date'], df_historical['pred_sharpe_full'], 
                 label='历史预测 (全概率)', color='royalblue', alpha=0.7, linewidth=2)
        ax2.plot(df_historical['date'], df_historical['pred_sharpe_top3'], 
                 label='历史预测 (Top-3)', color='darkviolet', linewidth=1.5, linestyle='--')
        ax2.plot(df_historical['date'], df_historical['pred_sharpe_top2'], 
                 label='历史预测 (Top-2)', color='purple', linewidth=1.5, linestyle='-.')
        ax2.plot(df_historical['date'], df_historical['pred_sharpe_top1'], 
                 label='历史预测 (Top-1)', color='red', linewidth=1.5, linestyle=':')
        ax2.plot(df_historical['date'], df_historical['actual_sharpe'], 
                 label='实际表现', color='darkorange', alpha=0.8, linewidth=2.5)
    
    # 未来预测线
    if not df_future.empty:
        ax2.plot(df_future['date'], df_future['pred_sharpe_full'], 
                 label='未来预测 (全概率)', color='lightblue', alpha=0.8, linewidth=2, linestyle='-')
        ax2.plot(df_future['date'], df_future['pred_sharpe_top3'], 
                 label='未来预测 (Top-3)', color='plum', linewidth=1.5, linestyle='--')
        ax2.plot(df_future['date'], df_future['pred_sharpe_top2'], 
                 label='未来预测 (Top-2)', color='mediumpurple', linewidth=1.5, linestyle='-.')
        ax2.plot(df_future['date'], df_future['pred_sharpe_top1'], 
                 label='未来预测 (Top-1)', color='lightcoral', linewidth=1.5, linestyle=':')
    
    # 分界线
    if not df_historical.empty and not df_future.empty:
        ax2.axvline(x=boundary_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # 添加训练/验证/测试集的背景色块
    if not df_historical.empty and 'dataset_type' in df_historical.columns:
        # 为每个数据集类型的连续区间添加背景
        for i in range(len(df_historical)):
            if i == 0 or df_historical.iloc[i]['dataset_type'] != df_historical.iloc[i-1]['dataset_type']:
                start_date = df_historical.iloc[i]['date']
                dataset_type = df_historical.iloc[i]['dataset_type']
                
                # 找到结束位置
                end_idx = i
                while end_idx < len(df_historical) - 1 and df_historical.iloc[end_idx + 1]['dataset_type'] == dataset_type:
                    end_idx += 1
                end_date = df_historical.iloc[end_idx]['date']
                
                # 根据类型设置颜色
                if dataset_type == 'train':
                    ax2.axvspan(start_date, end_date, alpha=0.1, color='blue')
                elif dataset_type == 'val':
                    ax2.axvspan(start_date, end_date, alpha=0.1, color='green')
                elif dataset_type == 'test':
                    ax2.axvspan(start_date, end_date, alpha=0.1, color='orange')
    
    ax2.set_title(f'{stock_code} - 未来20天夏普比率预测 vs 实际', fontsize=14, fontweight='bold')
    ax2.set_ylabel('夏普比率', fontsize=12)
    ax2.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 最大回撤对比
    ax3 = axes[2]
    # 历史预测线
    if not df_historical.empty:
        ax3.plot(df_historical['date'], df_historical['pred_drawdown_full'] * 100, 
                 label='历史预测 (全概率)', color='royalblue', alpha=0.7, linewidth=2)
        ax3.plot(df_historical['date'], df_historical['pred_drawdown_top3'] * 100, 
                 label='历史预测 (Top-3)', color='darkviolet', linewidth=1.5, linestyle='--')
        ax3.plot(df_historical['date'], df_historical['pred_drawdown_top2'] * 100, 
                 label='历史预测 (Top-2)', color='purple', linewidth=1.5, linestyle='-.')
        ax3.plot(df_historical['date'], df_historical['pred_drawdown_top1'] * 100, 
                 label='历史预测 (Top-1)', color='red', linewidth=1.5, linestyle=':')
        ax3.plot(df_historical['date'], df_historical['actual_drawdown'] * 100, 
                 label='实际表现', color='darkorange', alpha=0.8, linewidth=2.5)
    
    # 未来预测线
    if not df_future.empty:
        ax3.plot(df_future['date'], df_future['pred_drawdown_full'] * 100, 
                 label='未来预测 (全概率)', color='lightblue', alpha=0.8, linewidth=2, linestyle='-')
        ax3.plot(df_future['date'], df_future['pred_drawdown_top3'] * 100, 
                 label='未来预测 (Top-3)', color='plum', linewidth=1.5, linestyle='--')
        ax3.plot(df_future['date'], df_future['pred_drawdown_top2'] * 100, 
                 label='未来预测 (Top-2)', color='mediumpurple', linewidth=1.5, linestyle='-.')
        ax3.plot(df_future['date'], df_future['pred_drawdown_top1'] * 100, 
                 label='未来预测 (Top-1)', color='lightcoral', linewidth=1.5, linestyle=':')
    
    # 分界线
    if not df_historical.empty and not df_future.empty:
        ax3.axvline(x=boundary_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # 添加训练/验证/测试集的背景色块
    if not df_historical.empty and 'dataset_type' in df_historical.columns:
        # 为每个数据集类型的连续区间添加背景
        for i in range(len(df_historical)):
            if i == 0 or df_historical.iloc[i]['dataset_type'] != df_historical.iloc[i-1]['dataset_type']:
                start_date = df_historical.iloc[i]['date']
                dataset_type = df_historical.iloc[i]['dataset_type']
                
                # 找到结束位置
                end_idx = i
                while end_idx < len(df_historical) - 1 and df_historical.iloc[end_idx + 1]['dataset_type'] == dataset_type:
                    end_idx += 1
                end_date = df_historical.iloc[end_idx]['date']
                
                # 根据类型设置颜色
                if dataset_type == 'train':
                    ax3.axvspan(start_date, end_date, alpha=0.1, color='blue')
                elif dataset_type == 'val':
                    ax3.axvspan(start_date, end_date, alpha=0.1, color='green')
                elif dataset_type == 'test':
                    ax3.axvspan(start_date, end_date, alpha=0.1, color='orange')
    
    ax3.set_title(f'{stock_code} - 未来20天最大回撤预测 vs 实际', fontsize=14, fontweight='bold')
    ax3.set_ylabel('最大回撤 (%)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 设置x轴日期格式
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    # 总标题
    fig.suptitle(f'3D模型长期预测表现分析 - {model_name}\\n时间跨度: {years}年', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(config.MODEL_DIR, f"3d_long_term_analysis_{stock_code}_{timestamp}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"图表已保存至: {save_path}")
    
    # 计算并打印统计指标
    print_performance_statistics(df, stock_code, model_name)
    
    plt.show()
    
    return df, save_path

def print_performance_statistics(df, stock_code, model_name):
    """
    打印预测性能统计指标
    """
    print(f"\n{'='*80}")
    print(f"📊 {stock_code} - {model_name} 预测性能统计")
    print(f"{'='*80}")
    print(f"📅 分析期间: {df['date'].min().date()} 至 {df['date'].max().date()}")
    print(f"📝 样本数量: {len(df)} 个预测点")
    
    # 计算相关系数
    metrics = ['return', 'sharpe', 'drawdown']
    metric_names = ['收益率', '夏普比率', '最大回撤']
    
    for metric, name in zip(metrics, metric_names):
        actual_col = f'actual_{metric}'
        pred_full_col = f'pred_{metric}_full'
        pred_top3_col = f'pred_{metric}_top3'
        pred_top2_col = f'pred_{metric}_top2'
        pred_top1_col = f'pred_{metric}_top1'
        
        if actual_col in df.columns:
            # 只使用有实际数据的部分计算统计指标
            df_valid = df[df[actual_col].notna()]
            if not df_valid.empty:
                corr_full = df_valid[actual_col].corr(df_valid[pred_full_col])
                corr_top3 = df_valid[actual_col].corr(df_valid[pred_top3_col])
                corr_top2 = df_valid[actual_col].corr(df_valid[pred_top2_col])
                corr_top1 = df_valid[actual_col].corr(df_valid[pred_top1_col])
                
                rmse_full = np.sqrt(np.mean((df_valid[actual_col] - df_valid[pred_full_col])**2))
                rmse_top3 = np.sqrt(np.mean((df_valid[actual_col] - df_valid[pred_top3_col])**2))
                rmse_top2 = np.sqrt(np.mean((df_valid[actual_col] - df_valid[pred_top2_col])**2))
                rmse_top1 = np.sqrt(np.mean((df_valid[actual_col] - df_valid[pred_top1_col])**2))
            else:
                corr_full = corr_top3 = corr_top2 = corr_top1 = np.nan
                rmse_full = rmse_top3 = rmse_top2 = rmse_top1 = np.nan
            
            print(f"\n🎯 {name}:")
            print(f"  相关系数 - 全概率: {corr_full:.4f} | Top-3: {corr_top3:.4f} | Top-2: {corr_top2:.4f} | Top-1: {corr_top1:.4f}")
            print(f"  RMSE     - 全概率: {rmse_full:.4f} | Top-3: {rmse_top3:.4f} | Top-2: {rmse_top2:.4f} | Top-1: {rmse_top1:.4f}")
    
    print(f"{'='*80}")
    print("💡 说明:")
    print("  - 相关系数越接近1，预测方向准确性越高")
    print("  - RMSE越小，预测数值准确性越高") 
    print("  - 这些是基于AI模型对未来20天表现的预测分析")
    print(f"{'='*80}\n")

def draw_all_3d_models_comparison(stock_code, years=2):
    """
    绘制所有3D最佳模型的对比
    """
    # 查找所有3D最佳模型
    model_patterns = [
        "best_loss_top_*.pth",
        "best_*_acc.pth",
        "market_*_model.pth"
    ]
    
    model_paths = []
    for pattern in model_patterns:
        pattern_path = os.path.join(config.MODEL_DIR, pattern)
        model_paths.extend(glob.glob(pattern_path))
        
        # 也检查3d_models子目录
        subdir_pattern_path = os.path.join(config.MODEL_DIR, "3d_models", pattern)
        model_paths.extend(glob.glob(subdir_pattern_path))
    
    # 去重
    model_paths = list(set(model_paths))
    
    if not model_paths:
        logger.warning("未找到3D模型文件，使用默认模型路径")
        model_paths = [config.MODEL_PATH]
    
    logger.info(f"找到 {len(model_paths)} 个3D模型文件进行对比")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"正在分析模型: {os.path.basename(model_path)}")
            try:
                draw_3d_long_term_comparison(stock_code, model_path, years)
                print("\n" + "="*50 + "\n")
            except Exception as e:
                logger.error(f"分析模型 {model_path} 时出错: {e}")
                continue

def main():
    """
    主函数 - 交互式3D长期分析
    """
    print("🚀 3D模型长期预测vs实际表现分析工具")
    print("=" * 60)
    
    # 获取股票代码
    stock_code = input("请输入股票代码 (例: 002415): ").strip()
    if not stock_code:
        stock_code = "002415"  # 默认股票
    
    # 获取分析年数
    try:
        years = int(input("请输入分析年数 (默认3年): ") or "3")
    except ValueError:
        years = 3
    
    # 选择分析模式
    print("\n请选择分析模式:")
    print("1. 单个最佳模型分析")
    print("2. 所有模型对比分析")
    
    choice = input("请输入选择 (默认1): ").strip()
    
    try:
        if choice == "2":
            draw_all_3d_models_comparison(stock_code, years)
        else:
            df, save_path = draw_3d_long_term_comparison(stock_code, years=years)
            print(f"\n✅ 分析完成！图表已保存至: {save_path}")
            
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()