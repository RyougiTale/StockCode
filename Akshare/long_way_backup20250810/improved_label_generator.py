#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的3D软标签生成器 - 修复相对化指标问题
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
try:
    from . import config
    from .logger_config import get_logger
except ImportError:
    import config
    from logger_config import get_logger
from scipy.stats import percentileofscore
import os
import pickle

logger = get_logger(__name__)

class ImprovedRelativeMetricsCalculator:
    """
    改进的相对化指标计算器
    结合绝对指标和相对排名，提供更有区分度的标签
    """
    
    def __init__(self):
        self.stock_distributions = {}
        self.global_stats = None  # 全局统计量
        self.cache_file = os.path.join(config.MODEL_DIR, "improved_stock_distributions.pkl")
        
    def calculate_percentile_rank(self, value, distribution):
        """计算值在分布中的百分位排名"""
        if len(distribution) == 0:
            return 0.5
        return percentileofscore(distribution, value, kind='rank') / 100.0
    
    def fit_stock_distributions(self, stock_samples_dict):
        """为每只股票建立历史分布"""
        logger.info(f"开始为 {len(stock_samples_dict)} 只股票构建历史分布...")
        
        # 收集全局统计量
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        
        for stock_code, samples in stock_samples_dict.items():
            returns = []
            sharpes = []
            drawdowns = []
            
            for sample in samples:
                if 'future_prices' in sample and len(sample['future_prices']) > 1:
                    prices = pd.Series(sample['future_prices'])
                    metrics = self._calculate_raw_metrics(prices)
                    if metrics:
                        returns.append(metrics['total_return'])
                        sharpes.append(metrics['sharpe_ratio'])
                        drawdowns.append(metrics['max_drawdown'])
                        
                        # 收集全局数据
                        all_returns.append(metrics['total_return'])
                        all_sharpes.append(metrics['sharpe_ratio'])
                        all_drawdowns.append(metrics['max_drawdown'])
            
            if len(returns) > 10:  # 至少需要10个样本
                # 为每只股票存储分布信息和动态中心点
                self.stock_distributions[stock_code] = {
                    'total_return': {
                        'values': returns,
                        'mean': np.mean(returns),
                        'std': np.std(returns),
                        'quantiles': np.quantile(returns, [0.1, 0.25, 0.5, 0.75, 0.9])
                    },
                    'sharpe_ratio': {
                        'values': sharpes,
                        'mean': np.mean(sharpes),
                        'std': np.std(sharpes),
                        'quantiles': np.quantile(sharpes, [0.1, 0.25, 0.5, 0.75, 0.9])
                    },
                    'max_drawdown': {
                        'values': drawdowns,
                        'mean': np.mean(drawdowns),
                        'std': np.std(drawdowns),
                        'quantiles': np.quantile(drawdowns, [0.1, 0.25, 0.5, 0.75, 0.9])
                    }
                }
                
                logger.debug(f"股票 {stock_code} 分布构建完成，样本数: {len(returns)}")
                logger.debug(f"  回报率范围: [{min(returns):.4f}, {max(returns):.4f}]")
                logger.debug(f"  夏普比率范围: [{min(sharpes):.4f}, {max(sharpes):.4f}]")
                logger.debug(f"  最大回撤范围: [{min(drawdowns):.4f}, {max(drawdowns):.4f}]")
        
        # 构建全局统计量作为备选
        if all_returns:
            self.global_stats = {
                'total_return': {
                    'mean': np.mean(all_returns),
                    'std': np.std(all_returns),
                    'quantiles': np.quantile(all_returns, [0.1, 0.25, 0.5, 0.75, 0.9])
                },
                'sharpe_ratio': {
                    'mean': np.mean(all_sharpes),
                    'std': np.std(all_sharpes),
                    'quantiles': np.quantile(all_sharpes, [0.1, 0.25, 0.5, 0.75, 0.9])
                },
                'max_drawdown': {
                    'mean': np.mean(all_drawdowns),
                    'std': np.std(all_drawdowns),
                    'quantiles': np.quantile(all_drawdowns, [0.1, 0.25, 0.5, 0.75, 0.9])
                }
            }
            
            logger.info(f"全局统计量构建完成，总样本数: {len(all_returns)}")
            logger.debug("全局统计:")
            logger.debug(f"  回报率: 均值={self.global_stats['total_return']['mean']:.4f}, 标准差={self.global_stats['total_return']['std']:.4f}")
            logger.debug(f"  夏普比率: 均值={self.global_stats['sharpe_ratio']['mean']:.4f}, 标准差={self.global_stats['sharpe_ratio']['std']:.4f}")
            logger.debug(f"  最大回撤: 均值={self.global_stats['max_drawdown']['mean']:.4f}, 标准差={self.global_stats['max_drawdown']['std']:.4f}")
        
        # 保存分布到文件
        self.save_distributions()
        logger.info("所有股票分布构建完成")
    
    def _calculate_raw_metrics(self, price_series):
        """计算原始指标（与ThreeDimensionalLabelGenerator中的逻辑保持一致）"""
        if len(price_series) < 2:
            return None
        if (price_series <= 0).any():
            return None
            
        try:
            # 1. 总回报率
            total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
            
            # 2. 夏普比率
            if len(price_series) > 1:
                daily_returns = price_series.pct_change().dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
                    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # 3. 最大回撤
            cumulative_max = price_series.cummax()
            drawdown = (price_series - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown)
            }
            
        except Exception as e:
            logger.warning(f"计算原始指标时出错: {e}")
            return None
    
    def get_adaptive_centers(self, stock_code, metric_type):
        """
        获取自适应的标签中心点
        结合股票历史分布和绝对基准
        """
        # 基准中心点（绝对意义）
        baseline_centers = {
            'total_return': [-0.15, -0.05, 0.02, 0.08, 0.20],     # 基于市场表现的绝对基准
            'sharpe_ratio': [-1.0, 0.0, 0.5, 1.0, 2.0],          # 基于夏普比率的绝对基准  
            'max_drawdown': [-0.25, -0.15, -0.08, -0.04, -0.01]  # 基于回撤的绝对基准
        }
        
        # 如果有该股票的历史分布，进行自适应调整
        if stock_code in self.stock_distributions and metric_type in self.stock_distributions[stock_code]:
            stock_dist = self.stock_distributions[stock_code][metric_type]
            quantiles = stock_dist['quantiles']
            
            # 混合策略：70%权重给历史分布，30%权重给绝对基准
            baseline = np.array(baseline_centers[metric_type])
            relative = quantiles
            
            # 自适应混合
            adaptive_centers = 0.3 * baseline + 0.7 * relative
            return adaptive_centers.tolist()
        
        # 如果有全局统计量，使用全局基准
        elif self.global_stats and metric_type in self.global_stats:
            global_quantiles = self.global_stats[metric_type]['quantiles']
            baseline = np.array(baseline_centers[metric_type])
            
            # 50%全局分布 + 50%绝对基准
            adaptive_centers = 0.5 * baseline + 0.5 * global_quantiles
            return adaptive_centers.tolist()
        
        # 回退到绝对基准
        return baseline_centers[metric_type]
    
    def convert_to_relative(self, metrics, stock_code):
        """
        将绝对指标转换为相对化指标
        使用自适应中心点策略
        """
        if not metrics:
            return None
            
        relative_metrics = {}
        
        for metric_name in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            if metric_name in metrics:
                # 获取自适应中心点
                centers = self.get_adaptive_centers(stock_code, metric_name)
                
                # 计算相对位置
                value = metrics[metric_name]
                
                # 方法1: 基于中心点的相对位置（保持连续性）
                if value <= centers[0]:
                    relative_pos = 0.0
                elif value >= centers[-1]:
                    relative_pos = 1.0
                else:
                    # 在中心点之间插值
                    for i in range(len(centers) - 1):
                        if centers[i] <= value <= centers[i + 1]:
                            # 线性插值到[0, 1]区间的对应位置
                            progress = (value - centers[i]) / (centers[i + 1] - centers[i])
                            relative_pos = (i + progress) / (len(centers) - 1)
                            break
                    else:
                        relative_pos = 0.5  # 默认值
                
                relative_metrics[metric_name] = float(np.clip(relative_pos, 0.0, 1.0))
        
        return relative_metrics
        
    def save_distributions(self):
        """保存分布到文件"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data_to_save = {
                'stock_distributions': self.stock_distributions,
                'global_stats': self.global_stats
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info(f"股票分布已保存: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存股票分布失败: {e}")

class ImprovedThreeDimensionalLabelGenerator:
    """
    改进的3D软标签生成器
    修复了相对化指标的问题
    """
    
    def __init__(self, look_forward_days=20, temperature=0.1, use_relative_metrics=True):
        self.look_forward_days = look_forward_days
        self.temperature = temperature
        self.use_relative_metrics = use_relative_metrics
        
        if use_relative_metrics:
            logger.info("3D标签生成器初始化 (使用改进的相对化指标):")
            # 使用改进的相对化计算器
            self.relative_calculator = ImprovedRelativeMetricsCalculator()
        else:
            logger.info("3D标签生成器初始化 (使用绝对指标):")
            # 绝对指标使用固定的类别中心点
            self.return_centers = torch.tensor([-0.15, -0.05, 0.02, 0.08, 0.20], dtype=torch.float32)
            self.sharpe_centers = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
            self.drawdown_centers = torch.tensor([-0.25, -0.15, -0.08, -0.04, -0.01], dtype=torch.float32)
            
        logger.debug(f"  前瞻天数: {look_forward_days}")
        logger.debug(f"  温度参数: {temperature}")
        logger.debug(f"  使用相对化指标: {use_relative_metrics}")
    
    def fit_stock_distributions(self, stock_samples_dict):
        """为相对化指标构建股票分布"""
        if self.use_relative_metrics:
            self.relative_calculator.fit_stock_distributions(stock_samples_dict)
    
    def calculate_future_metrics(self, price_series):
        """计算未来N天的三个核心指标"""
        if len(price_series) < 2:
            return None
        if (price_series <= 0).any():
            return None
            
        # 1. 总回报率
        total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
        
        # 2. 夏普比率
        daily_returns = price_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 3. 最大回撤
        cumulative_max = price_series.cummax()
        drawdown = (price_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }
    
    def create_soft_label_3d(self, metrics, stock_code=None):
        """根据三个指标生成3D软标签"""
        if metrics is None:
            return {
                'return': torch.ones(5) / 5,
                'sharpe': torch.ones(5) / 5,
                'drawdown': torch.ones(5) / 5
            }
        
        if self.use_relative_metrics:
            # 使用改进的相对化指标
            relative_metrics = self.relative_calculator.convert_to_relative(metrics, stock_code)
            if relative_metrics is None:
                return {
                    'return': torch.ones(5) / 5,
                    'sharpe': torch.ones(5) / 5,
                    'drawdown': torch.ones(5) / 5
                }
            
            # 使用相对化后的指标值作为中心点计算软标签
            return_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
            sharpe_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
            drawdown_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
            
            return {
                'return': self._generate_soft_label(relative_metrics['total_return'], return_centers),
                'sharpe': self._generate_soft_label(relative_metrics['sharpe_ratio'], sharpe_centers),
                'drawdown': self._generate_soft_label(relative_metrics['max_drawdown'], drawdown_centers)
            }
        else:
            # 使用绝对指标
            return {
                'return': self._generate_soft_label(metrics['total_return'], self.return_centers),
                'sharpe': self._generate_soft_label(metrics['sharpe_ratio'], self.sharpe_centers),
                'drawdown': self._generate_soft_label(metrics['max_drawdown'], self.drawdown_centers)
            }
    
    def _generate_soft_label(self, value, centers):
        """基于距离和温度参数生成软标签"""
        # 计算与每个中心的距离
        distances = torch.abs(centers - value)
        
        # 使用温度参数控制软化程度
        logits = -distances / self.temperature
        
        # 转换为概率分布
        probabilities = F.softmax(logits, dim=0)
        
        return probabilities

# 向后兼容：使用改进的生成器
ThreeDimensionalLabelGenerator = ImprovedThreeDimensionalLabelGenerator

# 全局实例，用于整个训练过程
global_label_generator = None

def get_label_generator():
    """获取全局标签生成器实例"""
    global global_label_generator
    if global_label_generator is None:
        global_label_generator = ThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True  # 启用改进的相对化指标
        )
    return global_label_generator