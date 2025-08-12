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

class RelativeMetricsCalculator:
    """
    相对化指标计算器
    将每只股票的绝对指标转换为该股票历史数据中的相对位置(百分位排名)
    """
    
    def __init__(self):
        self.stock_distributions = {}  # 存储每只股票的历史分布
        self.cache_file = os.path.join(config.MODEL_DIR, "stock_distributions.pkl")
        self.load_cache()
    
    def fit_stock_distribution(self, stock_code, historical_metrics):
        """
        为每只股票建立历史指标分布
        
        Args:
            stock_code (str): 股票代码
            historical_metrics (list): 历史指标列表，每个元素包含total_return, sharpe_ratio, max_drawdown
        """
        if not historical_metrics:
            logger.warning(f"股票 {stock_code} 没有历史指标数据")
            return
        
        # 提取各个指标的值
        returns = [m['total_return'] for m in historical_metrics if m is not None]
        sharpes = [m['sharpe_ratio'] for m in historical_metrics if m is not None]  
        drawdowns = [m['max_drawdown'] for m in historical_metrics if m is not None]
        
        if len(returns) < 20:  # 至少需要20个数据点才有统计意义
            logger.warning(f"股票 {stock_code} 历史数据点不足: {len(returns)}")
            return
            
        self.stock_distributions[stock_code] = {
            'return_values': sorted(returns),
            'sharpe_values': sorted(sharpes),
            'drawdown_values': sorted(drawdowns),
            'sample_count': len(returns)
        }
        
        if config.DEBUG_MODE:
            logger.debug(f"股票 {stock_code} 分布建立完成，样本数: {len(returns)}")
            logger.debug(f"  回报率范围: [{min(returns):.4f}, {max(returns):.4f}]")
            logger.debug(f"  夏普比率范围: [{min(sharpes):.4f}, {max(sharpes):.4f}]")
            logger.debug(f"  最大回撤范围: [{min(drawdowns):.4f}, {max(drawdowns):.4f}]")
        else:
            logger.info(f"股票 {stock_code} 分布建立完成，样本数: {len(returns)}")
    
    def transform_to_relative(self, stock_code, current_metrics):
        """
        将当前指标转换为在该股票历史中的相对位置(0-1)
        
        Args:
            stock_code (str): 股票代码
            current_metrics (dict): 当前指标，包含total_return, sharpe_ratio, max_drawdown
            
        Returns:
            dict: 相对化指标，包含return_percentile, sharpe_percentile, drawdown_percentile
        """
        if stock_code not in self.stock_distributions:
            # 如果没有该股票的分布，返回0.5（中位数位置）
            logger.warning(f"股票 {stock_code} 没有历史分布，使用默认相对值")
            return {
                'return_percentile': 0.5,
                'sharpe_percentile': 0.5, 
                'drawdown_percentile': 0.5
            }
            
        dist = self.stock_distributions[stock_code]
        
        return {
            'return_percentile': self._get_percentile_rank(
                current_metrics['total_return'], dist['return_values']
            ),
            'sharpe_percentile': self._get_percentile_rank(
                current_metrics['sharpe_ratio'], dist['sharpe_values']
            ),
            'drawdown_percentile': self._get_percentile_rank(
                current_metrics['max_drawdown'], dist['drawdown_values']
            )
        }
    
    def _get_percentile_rank(self, value, sorted_values):
        """计算value在sorted_values中的百分位排名"""
        try:
            # 使用scipy的percentileofscore函数
            rank = percentileofscore(sorted_values, value, kind='rank') / 100.0
            # 确保在[0, 1]范围内
            return np.clip(rank, 0.0, 1.0)
        except:
            # 如果出错，返回中位数
            return 0.5
    
    def save_cache(self):
        """保存股票分布缓存到文件"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.stock_distributions, f)
            logger.info(f"股票分布缓存已保存: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存股票分布缓存失败: {e}")
    
    def load_cache(self):
        """从文件加载股票分布缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.stock_distributions = pickle.load(f)
                logger.info(f"已加载股票分布缓存: {len(self.stock_distributions)} 只股票")
            else:
                logger.info("未找到股票分布缓存文件，将重新计算")
        except Exception as e:
            logger.error(f"加载股票分布缓存失败: {e}")
            self.stock_distributions = {}
    
    def get_distribution_info(self, stock_code):
        """获取股票分布信息"""
        if stock_code in self.stock_distributions:
            return self.stock_distributions[stock_code]
        return None

class ThreeDimensionalLabelGenerator:
    """
    3D软标签生成器（改进版，支持相对化指标）
    生成三个维度的投资指标：回报率、夏普比率、最大回撤
    """
    
    def __init__(self, look_forward_days=20, temperature=0.1, use_relative_metrics=True):
        self.look_forward_days = look_forward_days
        self.temperature = temperature
        self.use_relative_metrics = use_relative_metrics
        
        if use_relative_metrics:
            logger.info("3D标签生成器初始化 (使用相对化指标):")
            # 相对化指标使用统一的百分位中心点
            self.return_centers = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
            self.sharpe_centers = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
            self.drawdown_centers = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
            
            # 初始化相对化计算器
            self.relative_calculator = RelativeMetricsCalculator()
            
        else:
            logger.info("3D标签生成器初始化 (使用绝对指标):")
            # 原来的绝对值中心点
            self.return_centers = torch.tensor([-0.10, -0.045, 0.0, 0.045, 0.10])
            self.sharpe_centers = torch.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])
            self.drawdown_centers = torch.tensor([-0.20, -0.10, -0.05, -0.02, 0.0])
            self.relative_calculator = None

        if config.DEBUG_MODE:
            logger.debug(f"  回报率中心: {self.return_centers.numpy()}")
            logger.debug(f"  夏普比率中心: {self.sharpe_centers.numpy()}")
            logger.debug(f"  最大回撤中心: {self.drawdown_centers.numpy()}")
        else:
            logger.info("3D标签生成器初始化完成")
    
    def fit_stock_distributions(self, stock_samples_dict):
        """
        为所有股票建立历史分布
        
        Args:
            stock_samples_dict (dict): {stock_code: [samples...]} 格式的字典
        """
        if not self.use_relative_metrics:
            logger.info("非相对化模式，跳过分布建立")
            return
            
        logger.info(f"开始为 {len(stock_samples_dict)} 只股票建立历史分布...")
        
        for stock_code, samples in stock_samples_dict.items():
            # 提取该股票所有样本的指标
            historical_metrics = []
            for sample in samples:
                if 'future_prices' in sample and len(sample['future_prices']) > 1:
                    future_prices = pd.Series(sample['future_prices'])
                    metrics = self.calculate_future_metrics(future_prices)
                    if metrics:
                        historical_metrics.append(metrics)
            
            # 为该股票建立分布
            if historical_metrics:
                self.relative_calculator.fit_stock_distribution(stock_code, historical_metrics)
        
        # 保存缓存
        self.relative_calculator.save_cache()
        logger.info("所有股票分布建立完成")
    
    def calculate_future_metrics(self, price_series):
        """
        计算未来N天的三个核心指标（不变）
        
        Args:
            price_series (pd.Series): 未来N天的价格序列
            
        Returns:
            dict: 包含三个指标的字典，如果数据不足则返回None
        """
        if len(price_series) < 2:
            return None
        if (price_series <= 0).any():
            return None
            
        # 1. 总回报率
        total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
        if np.isnan(total_return) or np.isinf(total_return):
            return None
            
        # 2. 夏普比率（简化版本，避免极端值）
        daily_returns = price_series.pct_change().dropna()
        if len(daily_returns) > 1:
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            
            if std_return > 1e-9:  # 避免除零
                sharpe_ratio = mean_return / std_return
                sharpe_ratio = np.clip(sharpe_ratio, -2.0, 2.0)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 3. 最大回撤
        cumulative_max = price_series.cummax()
        drawdown = (price_series - cumulative_max) / (cumulative_max + 1e-9)
        max_drawdown = drawdown.min()
        max_drawdown = np.clip(max_drawdown, -1.0, 0.0)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    
    def create_soft_label_3d(self, metrics, stock_code=None):
        """
        根据三个指标生成3D软标签（支持相对化）
        
        Args:
            metrics (dict): 包含三个指标的字典
            stock_code (str): 股票代码，用于相对化计算
            
        Returns:
            dict: 包含三个维度软标签的字典
        """
        if metrics is None:
            # 返回均匀分布
            return {
                'return': torch.ones(len(self.return_centers)) / len(self.return_centers),
                'sharpe': torch.ones(len(self.sharpe_centers)) / len(self.sharpe_centers),
                'drawdown': torch.ones(len(self.drawdown_centers)) / len(self.drawdown_centers)
            }
        
        if self.use_relative_metrics and stock_code and self.relative_calculator:
            # 转换为相对化指标
            relative_metrics = self.relative_calculator.transform_to_relative(stock_code, metrics)
            
            # 使用相对化指标生成软标签
            return_label = self._create_single_soft_label(
                relative_metrics['return_percentile'], self.return_centers
            )
            sharpe_label = self._create_single_soft_label(
                relative_metrics['sharpe_percentile'], self.sharpe_centers
            )
            drawdown_label = self._create_single_soft_label(
                relative_metrics['drawdown_percentile'], self.drawdown_centers
            )
        else:
            # 使用绝对指标生成软标签（原来的逻辑）
            return_label = self._create_single_soft_label(
                metrics['total_return'], self.return_centers
            )
            sharpe_label = self._create_single_soft_label(
                metrics['sharpe_ratio'], self.sharpe_centers
            )
            drawdown_label = self._create_single_soft_label(
                metrics['max_drawdown'], self.drawdown_centers
            )
        
        return {
            'return': return_label,
            'sharpe': sharpe_label,
            'drawdown': drawdown_label
        }
    
    def _create_single_soft_label(self, value, centers):
        """
        为单个维度创建软标签
        
        Args:
            value (float): 实际值
            centers (torch.Tensor): 类别中心点
            
        Returns:
            torch.Tensor: 软标签分布
        """
        # 计算与每个中心的距离
        distances = torch.abs(centers - value)
        
        # 使用温度参数控制软化程度
        logits = -distances / self.temperature
        
        # 转换为概率分布
        probabilities = F.softmax(logits, dim=0)
        
        return probabilities

# 全局实例，用于整个训练过程
global_label_generator = None

def get_label_generator():
    """获取全局标签生成器实例"""
    global global_label_generator
    if global_label_generator is None:
        global_label_generator = ThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True  # 启用相对化指标
        )
    return global_label_generator

def test_relative_metrics():
    """测试相对化指标功能"""
    logger.info("=== 测试相对化指标功能 ===")
    
    # 创建标签生成器
    generator = ThreeDimensionalLabelGenerator(use_relative_metrics=True)
    
    # 模拟股票数据
    stock_samples = {
        'TEST001': [],
        'TEST002': []
    }
    
    # 为TEST001生成一些样本（低波动股票）
    np.random.seed(42)
    for i in range(100):
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 20))
        stock_samples['TEST001'].append({
            'future_prices': prices
        })
    
    # 为TEST002生成一些样本（高波动股票）
    for i in range(100):
        prices = 10 * np.cumprod(1 + np.random.normal(0.002, 0.05, 20))
        stock_samples['TEST002'].append({
            'future_prices': prices
        })
    
    # 建立分布
    generator.fit_stock_distributions(stock_samples)
    
    # 测试相对化转换
    test_metrics = {
        'total_return': 0.05,
        'sharpe_ratio': 0.3,
        'max_drawdown': -0.02
    }
    
    for stock_code in ['TEST001', 'TEST002']:
        relative = generator.relative_calculator.transform_to_relative(stock_code, test_metrics)
        soft_label = generator.create_soft_label_3d(test_metrics, stock_code)
        
        logger.info(f"股票 {stock_code}:")
        logger.info(f"  相对化指标: {relative}")
        logger.info(f"  软标签示例: return={soft_label['return'].numpy()}")
    
    logger.info("相对化指标测试完成")

if __name__ == '__main__':
    test_relative_metrics()