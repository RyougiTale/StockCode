#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版 3D 软标签生成器（ImprovedThreeDimensionalLabelGenerator）

目的
- 为未来 N 日（LOOK_FORWARD_DAYS）的三类目标指标生成“软标签”（soft labels）：
  1) total_return（总回报率），2) sharpe_ratio（夏普比率），3) max_drawdown（最大回撤）。
- 使用“温度缩放的距离 softmax”得到 5 类概率分布，既保留连续性又便于分类训练。

核心思想与公式
- 总回报率: R = P_T / P_0 - 1
- 夏普（近似年化）: S = mean(r_d) / std(r_d) * sqrt(252)，r_d 为日收益率
- 最大回撤: MDD = min_t ((P_t - cummax(P))/cummax(P))
- 软标签: 对中心点向量 c = [c0..c4]，令 d_i = |x - c_i|，logit_i = - d_i / T
         p_i = exp(logit_i) / Σ_j exp(logit_j) = exp(-|x-c_i|/T) / Σ_j exp(-|x-c_j|/T)
  其中 T 为温度（temperature），T 越小，分布越“尖锐”。

两种标签策略
- 绝对模式：直接用固定“经验中心点”。
- 相对模式（默认）：先将绝对值映射到 [0,1] 的相对位置（基于该股/全局历史分布的分位点，分段线性插值），
  再对相对中心 [0, 0.25, 0.5, 0.75, 1.0] 生成软标签。
"""

from __future__ import annotations

import os  # 文件路径、目录操作
import pickle  # 序列化/反序列化分布缓存
from datetime import datetime  # 时间戳
from typing import Dict, List, Optional

import numpy as np  # 数值计算（均值、标准差、分位点等）
import pandas as pd  # 序列与向量化运算
import torch  # 张量运算
import torch.nn.functional as F  # softmax 等
from scipy.stats import percentileofscore  # 百分位排名

try:
    # 包内导入（模块作为包使用时）
    from . import config  # 包含 MODEL_DIR、SOFT_LABEL_CONFIG 等
    from .logger_config import get_logger  # 统一日志
except ImportError:  # 直接脚本运行的兼容导入
    import config
    from logger_config import get_logger


logger = get_logger(__name__)  # 模块级日志器


class ImprovedRelativeMetricsCalculator:
    """
    改进的相对化指标计算器。

    作用
    - 为每只股票构建目标指标（total_return / sharpe_ratio / max_drawdown）的历史分布：
      保存 values、mean、std、quantiles（分位点）。
    - 提供“自适应中心”计算：把“绝对基线中心”与“股票/全局分位点”做线性混合。
    - 提供“绝对值→相对位置 [0,1]”的分段线性插值映射。
    """

    def __init__(self) -> None:
        # 每只股票的分布统计：stock_distributions[stock_code][metric]
        self.stock_distributions: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        # 全局统计（所有股票汇总），用于回退和稳健性
        self.global_stats: Optional[Dict[str, Dict[str, float]]] = None
        # 分布缓存文件（避免重复统计）
        self.cache_file: str = os.path.join(config.MODEL_DIR, "improved_stock_distributions.pkl")

    def calculate_percentile_rank(self, value: float, distribution: List[float]) -> float:
        """计算 value 在 distribution 中的百分位（0~1）。

        - 若分布为空，返回 0.5 作为中性回退。
        - 使用 scipy 的 percentileofscore（kind='rank'）。
        """
        if len(distribution) == 0:
            return 0.5
        return percentileofscore(distribution, value, kind="rank") / 100.0

    def fit_stock_distributions(self, stock_samples_dict: Dict[str, List[dict]]) -> None:
        """根据样本为每只股票建立历史分布（带缓存优化）。

        参数
        - stock_samples_dict: {stock_code: [sample, ...]}
          每个 sample 需包含 'future_prices'（未来一段时间的价格序列）。
        """
        logger.info(f"开始为 {len(stock_samples_dict)} 只股票构建历史分布...")

        # 尝试加载缓存
        if self._load_distributions_cache():
            logger.info("使用缓存的股票分布")
            return

        logger.info("未找到有效缓存，重新计算股票分布...")
        
        # 全局统计容器（作为回退）
        all_returns: List[float] = []
        all_sharpes: List[float] = []
        all_drawdowns: List[float] = []

        # 向量化批量处理
        total_samples = sum(len(samples) for samples in stock_samples_dict.values())
        logger.info(f"总样本数: {total_samples}")

        for stock_code, samples in stock_samples_dict.items():
            logger.info(f"处理股票 {stock_code}: {len(samples)} 个样本")
            
            # 批量收集该股票的价格数据
            valid_prices = []
            for sample in samples:
                if "future_prices" in sample and len(sample["future_prices"]) > 1:
                    valid_prices.append(sample["future_prices"])
            
            if len(valid_prices) < 10:  # 样本太少，跳过
                continue
                
            # 批量计算该股票的指标
            returns, sharpes, drawdowns = self._batch_calculate_metrics_for_stock(valid_prices)
            
            # 累计到全局统计
            all_returns.extend(returns)
            all_sharpes.extend(sharpes)
            all_drawdowns.extend(drawdowns)

            # 至少需要一定样本量，避免噪声主导
            if len(returns) > 10:
                self.stock_distributions[stock_code] = {
                    "total_return": {
                        "values": returns,
                        "mean": float(np.mean(returns)),
                        "std": float(np.std(returns)),
                        "quantiles": np.quantile(returns, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                    },
                    "sharpe_ratio": {
                        "values": sharpes,
                        "mean": float(np.mean(sharpes)),
                        "std": float(np.std(sharpes)),
                        "quantiles": np.quantile(sharpes, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                    },
                    "max_drawdown": {
                        "values": drawdowns,
                        "mean": float(np.mean(drawdowns)),
                        "std": float(np.std(drawdowns)),
                        "quantiles": np.quantile(drawdowns, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                    },
                }

                logger.debug(
                    f"股票 {stock_code} 分布就绪，样本数: {len(returns)} | 回报[{min(returns):.4f},{max(returns):.4f}] | "
                    f"夏普[{min(sharpes):.4f},{max(sharpes):.4f}] | 回撤[{min(drawdowns):.4f},{max(drawdowns):.4f}]"
                )

        # 构建全局统计（用于数据不足时的回退）
        if all_returns:
            self.global_stats = {
                "total_return": {
                    "mean": float(np.mean(all_returns)),
                    "std": float(np.std(all_returns)),
                    "quantiles": np.quantile(all_returns, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                },
                "sharpe_ratio": {
                    "mean": float(np.mean(all_sharpes)),
                    "std": float(np.std(all_sharpes)),
                    "quantiles": np.quantile(all_sharpes, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                },
                "max_drawdown": {
                    "mean": float(np.mean(all_drawdowns)),
                    "std": float(np.std(all_drawdowns)),
                    "quantiles": np.quantile(all_drawdowns, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                },
            }

            logger.info(f"全局统计量构建完成，总样本数: {len(all_returns)}")
            logger.debug("全局统计:")
            logger.debug(f"  回报率: 均值={self.global_stats['total_return']['mean']:.4f}, 标准差={self.global_stats['total_return']['std']:.4f}")
            logger.debug(f"  夏普比率: 均值={self.global_stats['sharpe_ratio']['mean']:.4f}, 标准差={self.global_stats['sharpe_ratio']['std']:.4f}")
            logger.debug(f"  最大回撤: 均值={self.global_stats['max_drawdown']['mean']:.4f}, 标准差={self.global_stats['max_drawdown']['std']:.4f}")
        
        # 保存分布到文件
        self.save_distributions()
        self._save_distributions_cache()
        logger.info("所有股票分布构建完成")

    def _calculate_raw_metrics(self, price_series: pd.Series) -> Optional[Dict[str, float]]:
        """从价格序列计算三项原始指标。

        输入
        - price_series: 价格序列（如未来窗口的收盘价）。

        输出
        - dict: {'total_return': R, 'sharpe_ratio': S, 'max_drawdown': MDD}
        """
        if len(price_series) < 2:
            return None  # 样本太短
        if (price_series <= 0).any():
            return None  # 非法价格

        try:
            # 1) 总回报率 R
            total_return = float(price_series.iloc[-1] / price_series.iloc[0] - 1.0)

            # 2) 夏普（近似年化）。若样本方差极小，避免除零，回退 0.0
            if len(price_series) > 1:
                daily_returns = price_series.pct_change().dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
                    sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0

            # 3) 最大回撤 MDD
            cumulative_max = price_series.cummax()
            drawdown = (price_series - cumulative_max) / cumulative_max
            max_drawdown = float(drawdown.min())

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            }

        except Exception as e:
            logger.warning(f"计算原始指标时出错: {e}")
            return None

    def get_adaptive_centers(self, stock_code: str, metric_type: str) -> List[float]:
        """计算给定股票与指标的“自适应中心”。

        策略
        - 绝对基线中心（baseline）：经验阈值。
        - 若有“股票分位点”（quantiles），用 30% baseline + 70% quantiles。
        - 否则若有“全局分位点”，用 50% baseline + 50% global_quantiles。
        - 均无则回退 baseline。
        """
        # 基准（绝对）中心点（从小到大），长度固定为 5
        # 使用市场通用经验值，适用于A股市场大部分股票
        baseline_centers = {
            "total_return": [-0.10, -0.03, 0.00, 0.03, 0.10],  # 20天收益率的合理分布
            "sharpe_ratio": [-2.0, -0.5, 0.5, 1.5, 3.0],      # 年化夏普的典型范围
            "max_drawdown": [-0.20, -0.10, -0.06, -0.03, -0.01],  # 回撤的合理分布
        }

        # 股票级自适应（优先）
        if stock_code in self.stock_distributions and metric_type in self.stock_distributions[stock_code]:
            stock_dist = self.stock_distributions[stock_code][metric_type]
            quantiles = np.asarray(stock_dist["quantiles"], dtype=float)
            baseline = np.asarray(baseline_centers[metric_type], dtype=float)
            adaptive_centers = 0.3 * baseline + 0.7 * quantiles
            return adaptive_centers.tolist()

        # 全局级自适应（回退）
        elif self.global_stats and metric_type in self.global_stats:
            global_quantiles = np.asarray(self.global_stats[metric_type]["quantiles"], dtype=float)
            baseline = np.asarray(baseline_centers[metric_type], dtype=float)
            adaptive_centers = 0.5 * baseline + 0.5 * global_quantiles
            return adaptive_centers.tolist()

        # 最终回退：纯基线
        return baseline_centers[metric_type]

    def convert_to_relative(self, metrics: Dict[str, float], stock_code: str) -> Optional[Dict[str, float]]:
        """把绝对指标值映射为 [0,1] 的相对位置（分段线性插值）。

        算法
        - 获取自适应中心 centers = [c0..c4]（有序）。
        - x <= c0 -> 0；x >= c4 -> 1；
        - c_i <= x <= c_{i+1} -> progress=(x-c_i)/(c_{i+1}-c_i)，
          relative_pos=(i+progress)/(len(centers)-1)。
        - 最后 clip 到 [0,1]。
        """
        if not metrics:
            return None

        relative_metrics: Dict[str, float] = {}

        for metric_name in ["total_return", "sharpe_ratio", "max_drawdown"]:
            if metric_name in metrics:
                # 获取自适应中心点
                centers = self.get_adaptive_centers(stock_code, metric_name)
                # 计算相对位置
                value = float(metrics[metric_name])

                # 方法1: 基于中心点的相对位置（保持连续性）
                # # 边界处理
                if value <= centers[0]:
                    relative_pos = 0.0
                elif value >= centers[-1]:
                    relative_pos = 1.0
                else:
                    # 区间定位 + 线性插值到 [0,1]
                    relative_pos = 0.5  # 兜底
                    for i in range(len(centers) - 1):
                        left, right = centers[i], centers[i + 1]
                        if left <= value <= right and right > left:
                            progress = (value - left) / (right - left)
                            relative_pos = (i + progress) / (len(centers) - 1)
                            break

                relative_metrics[metric_name] = float(np.clip(relative_pos, 0.0, 1.0))

        return relative_metrics

    def save_distributions(self) -> None:
        """把分布统计缓存到本地文件，避免重复计算。"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data_to_save = {
                "stock_distributions": self.stock_distributions,
                "global_stats": self.global_stats,
            }
            with open(self.cache_file, "wb") as f:
                pickle.dump(data_to_save, f)
            logger.info(f"股票分布已保存: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存股票分布失败: {e}")
    
    def _batch_calculate_metrics_for_stock(self, valid_prices):
        """
        为单只股票批量计算指标（向量化优化）
        
        Args:
            valid_prices: 该股票的价格序列列表
            
        Returns:
            tuple: (returns, sharpes, drawdowns) 三个列表
        """
        returns = []
        sharpes = []
        drawdowns = []
        
        for prices in valid_prices:
            price_array = np.array(prices)
            
            if len(price_array) < 2 or (price_array <= 0).any():
                continue
                
            try:
                # 总回报率
                total_return = (price_array[-1] / price_array[0]) - 1
                
                # 夏普比率
                daily_returns = np.diff(price_array) / price_array[:-1]
                if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
                    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
                
                # 最大回撤
                cumulative_max = np.maximum.accumulate(price_array)
                drawdown = (price_array - cumulative_max) / cumulative_max
                max_drawdown = drawdown.min()
                
                returns.append(total_return)
                sharpes.append(sharpe_ratio)
                drawdowns.append(max_drawdown)
                
            except Exception as e:
                logger.warning(f"计算指标时出错: {e}")
                continue
        
        return returns, sharpes, drawdowns
    
    def _load_distributions_cache(self):
        """加载股票分布缓存"""
        try:
            cache_file = self.cache_file.replace('.pkl', '_optimized.pkl')
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 验证缓存版本  
            cache_hash = cached_data.get('config_hash')
            current_hash = self._get_cache_hash()
            
            if cache_hash == current_hash:
                self.stock_distributions = cached_data['stock_distributions']
                self.global_stats = cached_data.get('global_stats')
                return True
        except Exception as e:
            logger.debug(f"加载分布缓存失败: {e}")
        return False
    
    def _save_distributions_cache(self):
        """保存股票分布缓存"""
        try:
            cache_file = self.cache_file.replace('.pkl', '_optimized.pkl')
            cache_data = {
                'stock_distributions': self.stock_distributions,
                'global_stats': self.global_stats,
                'config_hash': self._get_cache_hash(),
                'created_time': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"股票分布缓存已保存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存分布缓存失败: {e}")
    
    def _get_cache_hash(self):
        """获取影响分布计算的配置哈希"""
        import hashlib
        
        config_items = [
            "look_forward_20",  # 固定值，因为这里在RelativeCalculator中没有直接访问
            "temperature_dynamic",  # 温度会在上层传入
            "relative_metrics_true", # 使用相对化指标
        ]
        
        config_str = str(config_items)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ImprovedThreeDimensionalLabelGenerator:
    """
    改进的 3D 软标签生成器。

    职责
    - 把三项真实指标转换为三组 5 维概率的软标签（相对/绝对两种模式）。
    - 可从价格序列计算目标指标（calculate_future_metrics）。

    关键参数
    - look_forward_days: 前瞻窗口长度（单位：交易日）。
    - temperature: 软标签温度 T（T 越小，概率越尖锐）。
    - use_relative_metrics: 是否使用相对化指标（默认 True）。
    """

    def __init__(self, look_forward_days: int = 20, temperature: float = 0.1, use_relative_metrics: bool = True) -> None:
        # 前瞻窗口 N（用于生成标签的未来区间）
        self.look_forward_days: int = look_forward_days
        # 软标签温度参数 T
        self.temperature: float = temperature
        # 是否启用相对化策略
        self.use_relative_metrics: bool = use_relative_metrics

        if use_relative_metrics:
            logger.info("3D 标签生成器初始化（使用相对化指标）")
            # 相对化组件：基于股票/全局历史分布，生成自适应中心并映射到相对位置
            self.relative_calculator = ImprovedRelativeMetricsCalculator()
        else:
            logger.info("3D 标签生成器初始化（使用绝对指标）")
            # 绝对中心（经验阈值），按从小到大排列
            self.return_centers = torch.tensor([-0.15, -0.05, 0.02, 0.08, 0.20], dtype=torch.float32)
            self.sharpe_centers = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
            self.drawdown_centers = torch.tensor([-0.25, -0.15, -0.08, -0.04, -0.01], dtype=torch.float32)

        logger.debug(f"前瞻天数: {look_forward_days}")
        logger.debug(f"温度参数: {temperature}")
        logger.debug(f"使用相对化: {use_relative_metrics}")

    def fit_stock_distributions(self, stock_samples_dict: Dict[str, List[dict]]) -> None:
        """在相对化模式下，为股票构建分布（供自适应中心与相对映射使用）。"""
        if self.use_relative_metrics:
            self.relative_calculator.fit_stock_distributions(stock_samples_dict)

    def calculate_future_metrics(self, price_series: pd.Series) -> Optional[Dict[str, float]]:
        """从价格序列计算未来 N 日的三项目标指标（供离线标注使用）。"""
        if len(price_series) < 2:
            return None
        if (price_series <= 0).any():
            return None

        # 1) 总回报率
        total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1.0

        # 2) 夏普（近似年化）
        daily_returns = price_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 3) 最大回撤
        cumulative_max = price_series.cummax()
        drawdown = (price_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
        }

    def create_soft_label_3d(self, metrics: Optional[Dict[str, float]], stock_code: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """根据真实指标生成 3D 软标签（每维 5 类概率）。

        返回
        - {'return': Tensor[5], 'sharpe': Tensor[5], 'drawdown': Tensor[5]}
        """
        # 缺失时使用均匀分布（信息最少假设）
        if metrics is None:
            return {
                "return": torch.ones(5, dtype=torch.float32) / 5,
                "sharpe": torch.ones(5, dtype=torch.float32) / 5,
                "drawdown": torch.ones(5, dtype=torch.float32) / 5,
            }

        if self.use_relative_metrics:
            # 相对化：把绝对三指标映射到 [0,1]
            relative_metrics = self.relative_calculator.convert_to_relative(metrics, stock_code or "")
            if relative_metrics is None:
                return {
                    "return": torch.ones(5, dtype=torch.float32) / 5,
                    "sharpe": torch.ones(5, dtype=torch.float32) / 5,
                    "drawdown": torch.ones(5, dtype=torch.float32) / 5,
                }

            # 使用统一的相对中心点
            return_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
            sharpe_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
            drawdown_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)

            return {
                "return": self._generate_soft_label(float(relative_metrics["total_return"]), return_centers),
                "sharpe": self._generate_soft_label(float(relative_metrics["sharpe_ratio"]), sharpe_centers),
                "drawdown": self._generate_soft_label(float(relative_metrics["max_drawdown"]), drawdown_centers),
            }

        # 绝对模式：直接使用绝对中心
        return {
            "return": self._generate_soft_label(float(metrics["total_return"]), self.return_centers),
            "sharpe": self._generate_soft_label(float(metrics["sharpe_ratio"]), self.sharpe_centers),
            "drawdown": self._generate_soft_label(float(metrics["max_drawdown"]), self.drawdown_centers),
        }

    def _generate_soft_label(self, value: float, centers: torch.Tensor) -> torch.Tensor:
        """根据“温度缩放的距离 softmax”生成 5 维概率。

        输入
        - value: 标量（绝对值或相对值）。
        - centers: 形如 [c0..c4] 的 1D 张量。
        """
        # 计算到每个中心点的绝对距离 d_i
        distances = torch.abs(centers - float(value))
        # 使用温度参数控制软化程度
        # 缩放为 logits（负号使得“更近 -> logit 更大”）
        logits = -distances / float(self.temperature)
        # 沿维度 0 softmax，得到概率分布
        probabilities = F.softmax(logits, dim=0)
        return probabilities


# 兼容性别名：外部可用 ThreeDimensionalLabelGenerator 名称访问改进实现
ThreeDimensionalLabelGenerator = ImprovedThreeDimensionalLabelGenerator


# 全局实例（训练/推理可共享，避免重复初始化）
global_label_generator: Optional[ImprovedThreeDimensionalLabelGenerator] = None


def get_label_generator() -> ImprovedThreeDimensionalLabelGenerator:
    """获取/缓存一个全局的标签生成器实例。

    默认启用相对化策略；温度与前瞻天数取自配置项 `SOFT_LABEL_CONFIG`。
    """
    global global_label_generator
    if global_label_generator is None:
        global_label_generator = ThreeDimensionalLabelGenerator(
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True,
        )
    return global_label_generator

