#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资决策可视化工具 - 基于3D模型预测结果
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

# 导入long_way模块
try:
    from . import config
    from .model_3d import create_3d_model
    from .data_utils import resample_to_period, calculate_features
    from .rolling_scaler import RollingWindowScaler
    from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    from .logger_config import get_logger
except ImportError:
    import config
    from model_3d import create_3d_model
    from data_utils import resample_to_period, calculate_features
    from rolling_scaler import RollingWindowScaler
    from improved_label_generator import ImprovedThreeDimensionalLabelGenerator
    from logger_config import get_logger

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__)

class InvestmentDecisionVisualizer:
    """投资决策可视化器"""
    
    def __init__(self, model_path=None):
        """
        初始化可视化器
        
        Args:
            model_path: 模型路径，默认使用best_loss_top_1.pth
        """
        self.model_path = model_path or self._find_best_model()
        self.model = None
        self.label_generator = ImprovedThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        
        # 设置数据处理器
        self.daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
        self.weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
        self.monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)
        
        self._load_model()
    
    def _find_best_model(self):
        """查找最佳模型文件"""
        possible_paths = [
            os.path.join(config.MODEL_DIR, "best_loss_top.pth"),
            os.path.join(config.MODEL_DIR, "3d_models", "best_loss_top.pth"),
            config.MODEL_PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到模型文件: {path}")
                return path
        
        raise FileNotFoundError("未找到可用的模型文件")
    
    def _load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.model_path}")
        self.model = create_3d_model(config).to(config.DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=config.DEVICE))
        self.model.eval()
        logger.info("模型加载完成")
    
    def _prepare_data(self, stock_code, days=30):
        """
        准备股票数据
        
        Args:
            stock_code: 股票代码
            days: 分析天数
            
        Returns:
            准备好的数据
        """
        logger.info(f"正在准备 {stock_code} 的数据...")
        
        # 读取历史数据
        daily_df = read_history_by_code(stock_code)
        if daily_df is None or daily_df.empty:
            raise ValueError(f"无法获取股票 {stock_code} 的数据")
        
        # 特征工程
        daily_featured = calculate_features(daily_df.copy(), 'daily')
        weekly_featured = calculate_features(resample_to_period(daily_df.copy(), 'W-FRI'), 'weekly')
        monthly_featured = calculate_features(resample_to_period(daily_df.copy(), 'ME'), 'monthly')
        
        # 数据清洗
        for col in config.FEATURE_COLUMNS['daily']:
            if col in daily_featured.columns:
                daily_featured[col] = pd.to_numeric(daily_featured[col], errors='coerce')
        
        for col in config.FEATURE_COLUMNS['weekly']:
            if col in weekly_featured.columns:
                weekly_featured[col] = pd.to_numeric(weekly_featured[col], errors='coerce')
        
        for col in config.FEATURE_COLUMNS['monthly']:
            if col in monthly_featured.columns:
                monthly_featured[col] = pd.to_numeric(monthly_featured[col], errors='coerce')
        
        # 归一化
        daily_featured = self.daily_scaler.fit_transform(daily_featured, config.FEATURE_COLUMNS['daily'])
        weekly_featured = self.weekly_scaler.fit_transform(weekly_featured, config.FEATURE_COLUMNS['weekly'])
        monthly_featured = self.monthly_scaler.fit_transform(monthly_featured, config.FEATURE_COLUMNS['monthly'])
        
        # 获取最近N天的数据
        end_date = daily_df['date'].max()
        start_date = end_date - pd.Timedelta(days=days)
        target_df = daily_df[daily_df['date'] >= start_date].copy()
        
        return daily_featured, weekly_featured, monthly_featured, target_df, daily_df
    
    def predict_stock(self, stock_code, days=30):
        """
        对股票进行预测
        
        Args:
            stock_code: 股票代码
            days: 预测天数
            
        Returns:
            预测结果
        """
        daily_featured, weekly_featured, monthly_featured, target_df, full_df = self._prepare_data(stock_code, days)
        
        predictions = []
        
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
                output = self.model(daily_tensor, weekly_tensor, monthly_tensor)
                
                # 提取3D预测结果
                return_probs = torch.softmax(output['return'], dim=1).cpu().numpy()[0]
                sharpe_probs = torch.softmax(output['sharpe'], dim=1).cpu().numpy()[0] 
                drawdown_probs = torch.softmax(output['drawdown'], dim=1).cpu().numpy()[0]
                
                # 计算期望值（用于投资决策）
                return_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                sharpe_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                drawdown_centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                
                expected_return = np.sum(return_probs * return_centers)
                expected_sharpe = np.sum(sharpe_probs * sharpe_centers)
                expected_drawdown = np.sum(drawdown_probs * drawdown_centers)
                
                # 计算置信度（最高概率）
                return_confidence = np.max(return_probs)
                sharpe_confidence = np.max(sharpe_probs)
                drawdown_confidence = np.max(drawdown_probs)
                
                predictions.append({
                    'date': current_date,
                    'close_price': row['close'],
                    'return_probs': return_probs,
                    'sharpe_probs': sharpe_probs,
                    'drawdown_probs': drawdown_probs,
                    'expected_return': expected_return,
                    'expected_sharpe': expected_sharpe,
                    'expected_drawdown': expected_drawdown,
                    'return_confidence': return_confidence,
                    'sharpe_confidence': sharpe_confidence,
                    'drawdown_confidence': drawdown_confidence,
                    'overall_confidence': (return_confidence + sharpe_confidence + drawdown_confidence) / 3
                })
        
        return predictions, full_df
    
    def create_investment_dashboard(self, stock_code, days=30, save_path=None):
        """
        创建投资决策仪表板
        
        Args:
            stock_code: 股票代码
            days: 分析天数
            save_path: 保存路径
        """
        logger.info(f"正在创建 {stock_code} 的投资决策仪表板...")
        
        # 获取预测结果
        predictions, full_df = self.predict_stock(stock_code, days)
        
        if not predictions:
            logger.error("没有获取到预测结果")
            return
        
        # 转换为DataFrame便于处理
        pred_df = pd.DataFrame(predictions)
        
        # 创建图表
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'{stock_code} 价格走势与预测信心度',
                '3D预测分布热力图',
                '预期收益率时间序列',
                '预期夏普比率时间序列', 
                '预期最大回撤时间序列',
                '投资决策综合评分'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "heatmap"}],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.08
        )
        
        # 1. 价格走势与预测信心度
        fig.add_trace(
            go.Scatter(
                x=pred_df['date'],
                y=pred_df['close_price'],
                name='收盘价',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_df['date'],
                y=pred_df['overall_confidence'],
                name='预测信心度',
                line=dict(color='red', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # 2. 3D预测分布热力图（最新一天）
        latest_pred = predictions[-1]
        heatmap_data = np.array([
            latest_pred['return_probs'],
            latest_pred['sharpe_probs'], 
            latest_pred['drawdown_probs']
        ])
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=['很差', '较差', '一般', '较好', '很好'],
                y=['收益率', '夏普比率', '最大回撤'],
                colorscale='RdYlGn',
                text=np.round(heatmap_data, 3),
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=1, col=2
        )
        
        # 3. 预期收益率时间序列
        fig.add_trace(
            go.Scatter(
                x=pred_df['date'],
                y=pred_df['expected_return'],
                name='预期收益率',
                line=dict(color='green', width=3),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # 4. 预期夏普比率时间序列  
        fig.add_trace(
            go.Scatter(
                x=pred_df['date'],
                y=pred_df['expected_sharpe'],
                name='预期夏普比率',
                line=dict(color='orange', width=3)
            ),
            row=2, col=1
        )
        
        # 5. 预期最大回撤时间序列
        fig.add_trace(
            go.Scatter(
                x=pred_df['date'],
                y=pred_df['expected_drawdown'],
                name='预期最大回撤',
                line=dict(color='purple', width=3)
            ),
            row=3, col=1
        )
        
        # 生成投资建议
        investment_advice = self._generate_investment_advice(predictions)
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title=f'{stock_code} 投资决策分析仪表板<br><sub>{investment_advice}</sub>',
            title_font_size=20,
            showlegend=True
        )
        
        # 显示图表
        fig.show()
        
        # 保存图表
        if save_path:
            fig.write_html(save_path)
            logger.info(f"仪表板已保存至: {save_path}")
        
        # 打印详细投资建议
        self._print_investment_summary(stock_code, predictions)
        
        return fig
    
    def _generate_investment_advice(self, predictions):
        """生成投资建议"""
        if not predictions:
            return "数据不足，无法给出建议"
        
        latest = predictions[-1]
        recent_avg = np.mean([p['expected_return'] for p in predictions[-5:]])  # 最近5天平均
        
        # 综合评分
        score = (
            latest['expected_return'] * 0.4 +
            latest['expected_sharpe'] * 0.3 +
            (1 - latest['expected_drawdown']) * 0.2 +
            latest['overall_confidence'] * 0.1
        )
        
        if score > 0.7:
            return "🟢 强烈建议买入 - 预期表现优异"
        elif score > 0.5:
            return "🟡 建议买入 - 预期表现良好"
        elif score > 0.3:
            return "🟠 谨慎观望 - 预期表现一般"
        else:
            return "🔴 建议避免 - 预期表现较差"
    
    def _print_investment_summary(self, stock_code, predictions):
        """打印投资总结"""
        if not predictions:
            return
        
        latest = predictions[-1]
        
        print(f"\n{'='*60}")
        print(f"📊 {stock_code} 投资决策分析报告")
        print(f"{'='*60}")
        print(f"📅 分析日期: {latest['date'].strftime('%Y-%m-%d')}")
        print(f"💰 当前价格: ¥{latest['close_price']:.2f}")
        print(f"\n🎯 预测指标:")
        print(f"  📈 预期收益率评分: {latest['expected_return']:.3f} (信心度: {latest['return_confidence']:.3f})")
        print(f"  📊 预期夏普比率评分: {latest['expected_sharpe']:.3f} (信心度: {latest['sharpe_confidence']:.3f})")  
        print(f"  📉 预期回撤风险评分: {latest['expected_drawdown']:.3f} (信心度: {latest['drawdown_confidence']:.3f})")
        print(f"  🎲 综合预测信心度: {latest['overall_confidence']:.3f}")
        
        # 计算趋势
        if len(predictions) > 1:
            trend_return = latest['expected_return'] - predictions[-2]['expected_return']
            trend_symbol = "📈" if trend_return > 0 else "📉" if trend_return < 0 else "➡️"
            print(f"  📊 收益预期趋势: {trend_symbol} {trend_return:+.3f}")
        
        # 投资建议
        advice = self._generate_investment_advice(predictions)
        print(f"\n💡 投资建议: {advice}")
        
        # 风险提示
        print(f"\n⚠️  风险提示:")
        print(f"  - 此分析基于历史数据和AI模型，不构成投资建议")
        print(f"  - 股市有风险，投资需谨慎")
        print(f"  - 建议结合基本面分析和其他技术指标")
        print(f"{'='*60}")

def main():
    """主函数 - 交互式投资决策分析"""
    print("🚀 AI投资决策助手")
    print("=" * 50)
    
    # 获取股票代码
    stock_code = input("请输入股票代码 (例: 002415): ").strip()
    if not stock_code:
        stock_code = "002415"  # 默认股票
    
    # 获取分析天数
    try:
        days = int(input("请输入分析天数 (默认30天): ") or "30")
    except ValueError:
        days = 30
    
    try:
        # 创建可视化器
        visualizer = InvestmentDecisionVisualizer()
        
        # 生成仪表板
        save_path = f"{stock_code}_investment_dashboard.html"
        visualizer.create_investment_dashboard(stock_code, days, save_path)
        
        print(f"\n✅ 分析完成！仪表板已保存为: {save_path}")
        print("请在浏览器中打开HTML文件查看完整的交互式仪表板")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()