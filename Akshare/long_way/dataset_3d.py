import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
try:
    from .improved_label_generator import ImprovedThreeDimensionalLabelGenerator as ThreeDimensionalLabelGenerator
    from . import config
    from .logger_config import get_logger
except ImportError:
    from improved_label_generator import ImprovedThreeDimensionalLabelGenerator as ThreeDimensionalLabelGenerator
    import config
    from logger_config import get_logger

# 获取日志记录器
logger = get_logger(__name__)

class Market3DClassificationDataset(Dataset):
    """
    3D市场分类数据集（支持相对化指标）
    为每个样本生成三维软标签：回报率、夏普比率、最大回撤
    """
    
    def __init__(self, samples, look_forward_days=20, temperature=0.1, use_relative_metrics=True, stock_distributions=None):
        """
        Args:
            samples (list of dict): 样本列表，每个样本包含daily、weekly、monthly数据和future_prices
            look_forward_days (int): 前瞻天数
            temperature (float): 软标签温度参数
            use_relative_metrics (bool): 是否使用相对化指标
            stock_distributions (dict): 预计算的股票分布信息
        """
        self.samples = samples
        self.use_relative_metrics = use_relative_metrics
        
        # 创建3D标签生成器
        self.label_generator = ThreeDimensionalLabelGenerator(
            look_forward_days=look_forward_days,
            temperature=temperature,
            use_relative_metrics=use_relative_metrics
        )
        
        if use_relative_metrics and stock_distributions:
            # 如果提供了预计算的分布，直接使用
            self.label_generator.relative_calculator.stock_distributions = stock_distributions
            logger.info(f"使用预计算的股票分布: {len(stock_distributions)} 只股票")
        elif use_relative_metrics:
            # 否则需要从样本计算分布
            self._build_distributions_from_samples()
        
        logger.info("开始预计算3D软标签...")
        logger.info(f"处理 {len(self.samples)} 个样本...")
        
        # 批量优化：预先收集所有future_prices和相关信息
        batch_future_prices = []
        batch_stock_codes = []
        valid_indices = []
        
        logger.info("收集样本数据...")
        for i, sample in enumerate(self.samples):
            if 'future_prices' in sample and len(sample['future_prices']) > 1:
                batch_future_prices.append(sample['future_prices'])
                batch_stock_codes.append(sample.get('stock_code', None))
                valid_indices.append(i)
        
        logger.info(f"有效样本数: {len(batch_future_prices)}")
        
        # 批量计算指标
        logger.info("批量计算指标...")
        batch_metrics = self._batch_calculate_metrics(batch_future_prices)
        
        # 批量生成软标签
        logger.info("批量生成软标签...")
        batch_labels = self._batch_generate_soft_labels(batch_metrics, batch_stock_codes)
        
        # 组装结果
        logger.info("组装最终结果...")
        self.soft_labels_3d = []
        self.raw_metrics = []
        invalid_count = 0
        
        # 转换为集合提升查找性能
        valid_indices_set = set(valid_indices)
        
        batch_idx = 0
        for i in range(len(self.samples)):
            if i in valid_indices_set:
                # 使用批量计算的结果
                self.raw_metrics.append(batch_metrics[batch_idx])
                if self._validate_3d_label(batch_labels[batch_idx]):
                    self.soft_labels_3d.append(batch_labels[batch_idx])
                else:
                    self.soft_labels_3d.append({
                        'return': torch.ones(5) / 5,
                        'sharpe': torch.ones(5) / 5,
                        'drawdown': torch.ones(5) / 5
                    })
                    invalid_count += 1
                batch_idx += 1
            else:
                # 无效样本，使用默认值
                self.raw_metrics.append(None)
                self.soft_labels_3d.append({
                    'return': torch.ones(5) / 5,
                    'sharpe': torch.ones(5) / 5,
                    'drawdown': torch.ones(5) / 5
                })
                invalid_count += 1
        
        logger.info(f"3D软标签预计算完成: 总样本数={len(self.samples)}, 无效标签数={invalid_count}, 有效率={(len(self.samples) - invalid_count) / len(self.samples) * 100:.1f}%")
        
        # 分析标签分布
        self._analyze_label_distribution()
    
    def _build_distributions_from_samples(self):
        """从样本中构建股票分布"""
        logger.info("从样本构建股票历史分布...")
        
        # 按股票代码分组样本
        stock_samples_dict = {}
        for sample in self.samples:
            stock_code = sample.get('stock_code', 'UNKNOWN')
            if stock_code not in stock_samples_dict:
                stock_samples_dict[stock_code] = []
            stock_samples_dict[stock_code].append(sample)
        
        # 为每只股票建立分布
        self.label_generator.fit_stock_distributions(stock_samples_dict)

    def _batch_calculate_metrics(self, batch_future_prices):
        """
        批量计算指标，大幅提升性能
        
        Args:
            batch_future_prices (list): 批量的future_prices数据
            
        Returns:
            list: 批量计算的指标结果
        """
        batch_metrics = []
        
        # 分批处理，避免内存问题
        batch_size = 1000
        total_batches = (len(batch_future_prices) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            if batch_idx % 10 == 0:  # 每10批输出一次进度
                logger.info(f"  计算指标进度: {batch_idx + 1}/{total_batches}")
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(batch_future_prices))
            current_batch = batch_future_prices[start_idx:end_idx]
            
            # 批量处理当前批次
            for future_prices in current_batch:
                # 直接在numpy数组上计算，避免pandas转换开销
                price_array = np.array(future_prices)
                
                if len(price_array) < 2 or (price_array <= 0).any():
                    batch_metrics.append(None)
                    continue
                
                # 向量化计算三个指标
                try:
                    # 1. 总回报率
                    total_return = (price_array[-1] / price_array[0]) - 1
                    
                    # 2. 夏普比率 - 向量化计算
                    if len(price_array) > 1:
                        daily_returns = np.diff(price_array) / price_array[:-1]
                        if len(daily_returns) > 0 and daily_returns.std() > 1e-8:
                            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                        else:
                            sharpe_ratio = 0.0
                    else:
                        sharpe_ratio = 0.0
                    
                    # 3. 最大回撤 - 向量化计算
                    cumulative_max = np.maximum.accumulate(price_array)
                    drawdown = (price_array - cumulative_max) / cumulative_max
                    max_drawdown = drawdown.min()
                    
                    metrics = {
                        'total_return': float(total_return),
                        'sharpe_ratio': float(sharpe_ratio),
                        'max_drawdown': float(max_drawdown)
                    }
                    batch_metrics.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"计算指标时出错: {e}")
                    batch_metrics.append(None)
        
        return batch_metrics
    
    def _batch_generate_soft_labels(self, batch_metrics, batch_stock_codes):
        """
        批量生成软标签
        
        Args:
            batch_metrics (list): 批量计算的指标
            batch_stock_codes (list): 对应的股票代码
            
        Returns:
            list: 批量生成的软标签
        """
        batch_labels = []
        
        # 分批处理
        batch_size = 1000
        total_batches = (len(batch_metrics) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            if batch_idx % 10 == 0:  # 每10批输出一次进度
                logger.info(f"  生成标签进度: {batch_idx + 1}/{total_batches}")
                
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(batch_metrics))
            
            for i in range(start_idx, end_idx):
                metrics = batch_metrics[i]
                stock_code = batch_stock_codes[i]
                
                if metrics is not None:
                    try:
                        label_3d = self.label_generator.create_soft_label_3d(metrics, stock_code)
                        batch_labels.append(label_3d)
                    except Exception as e:
                        logger.warning(f"生成软标签时出错: {e}")
                        batch_labels.append({
                            'return': torch.ones(5) / 5,
                            'sharpe': torch.ones(5) / 5,
                            'drawdown': torch.ones(5) / 5
                        })
                else:
                    batch_labels.append({
                        'return': torch.ones(5) / 5,
                        'sharpe': torch.ones(5) / 5,
                        'drawdown': torch.ones(5) / 5
                    })
        
        return batch_labels
    
    def _validate_3d_label(self, label_3d):
        """验证3D标签的有效性"""
        for key, label in label_3d.items():
            if torch.isnan(label).any() or torch.isinf(label).any():
                return False
            # 放宽最小值检查：只要不是负数即可，允许极小的概率值
            if (label < 0).any():
                return False
            # 放宽概率和检查：允许更大的数值误差
            if abs(label.sum().item() - 1.0) > 1e-4:
                return False
        return True
    
    def _analyze_label_distribution(self):
        """分析标签分布和底层指标分布"""
        if not config.DEBUG_MODE:
            return  # 非调试模式下跳过详细分析
            
        logger.debug("\n--- 3D标签分布分析 ---")

        # 收集所有真实计算出的指标值
        all_metrics = [m for m in self.raw_metrics if m is not None]
        if not all_metrics:
            logger.debug("没有有效的指标可供分析。")
            return

        df_metrics = pd.DataFrame(all_metrics)

        logger.debug("\n--- 指标真实值分布分析 ---")
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        for dim in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            if dim in df_metrics.columns:
                percentiles = df_metrics[dim].quantile(quantiles).values
                logger.debug(f"【{dim.upper()}】分位数:")
                logger.debug(f"  {np.round(percentiles, 4)}")

        # 统计每个维度的软标签分布
        for dim in ['return', 'sharpe', 'drawdown']:
            # 收集所有标签
            all_labels = torch.stack([label[dim] for label in self.soft_labels_3d])
            
            # 计算期望值（加权平均）
            # 为改进的标签生成器提供兼容性
            if hasattr(self.label_generator, 'return_centers'):
                if dim == 'return':
                    centers = self.label_generator.return_centers
                elif dim == 'sharpe':
                    centers = self.label_generator.sharpe_centers
                else:  # drawdown
                    centers = self.label_generator.drawdown_centers
                
                expected_values = (all_labels * centers.unsqueeze(0)).sum(dim=1)
            else:
                # 使用改进的标签生成器时，使用固定的中心点
                centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
                expected_values = (all_labels * centers.unsqueeze(0)).sum(dim=1)
            
            logger.debug(f"\n{dim.upper()} 维度:")
            logger.debug(f"  期望值范围: [{expected_values.min().item():.4f}, {expected_values.max().item():.4f}]")
            logger.debug(f"  期望值均值: {expected_values.mean().item():.4f}")
            logger.debug(f"  期望值标准差: {expected_values.std().item():.4f}")
            
            # 统计最可能的类别
            most_likely_classes = all_labels.argmax(dim=1)
            class_counts = torch.bincount(most_likely_classes, minlength=5)
            logger.debug(f"  类别分布: {class_counts.numpy()}")
    
    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        根据索引获取一个样本
        
        Returns:
            dict: 包含输入数据和3D标签的字典
        """
        sample = self.samples[idx]
        
        # 将输入数据转换为tensor
        daily_data = torch.from_numpy(sample['daily'].astype(np.float32))
        weekly_data = torch.from_numpy(sample['weekly'].astype(np.float32))
        monthly_data = torch.from_numpy(sample['monthly'].astype(np.float32))
        
        # 获取3D软标签
        label_3d = self.soft_labels_3d[idx]
        
        return {
            'daily': daily_data,
            'weekly': weekly_data,
            'monthly': monthly_data,
            'labels_3d': label_3d
            # 移除date字段，因为pandas Timestamp无法被DataLoader处理
        }
    
    def get_label_statistics(self):
        """
        获取标签统计信息
        """
        stats = {}
        
        for dim in ['return', 'sharpe', 'drawdown']:
            all_labels = torch.stack([label[dim] for label in self.soft_labels_3d])
            
            # 计算熵（衡量不确定性）
            entropy = -(all_labels * torch.log(all_labels + 1e-8)).sum(dim=1)
            
            # 计算最大概率（衡量置信度）
            max_probs = all_labels.max(dim=1)[0]
            
            stats[dim] = {
                'mean_entropy': entropy.mean().item(),
                'std_entropy': entropy.std().item(),
                'mean_confidence': max_probs.mean().item(),
                'std_confidence': max_probs.std().item()
            }
        
        return stats

def split_samples_by_market_periods(all_samples):
    """
    按市场时期切分样本数据
    
    Args:
        all_samples (list): 所有样本（按时间排序）
        
    Returns:
        dict: {"recent": samples, "middle": samples, "distant": samples}
    """
    from datetime import datetime, timedelta
    
    # 获取配置
    period_config = config.MARKET_PERIOD_CONFIG
    
    if not period_config["enable_period_split"]:
        logger.info("市场时期切分已禁用，使用全部数据")
        return {"all": all_samples}
    
    # 计算时间边界
    current_time = datetime.now()
    recent_cutoff = current_time - timedelta(days=365 * period_config["recent_years"])
    middle_cutoff = current_time - timedelta(days=365 * period_config["middle_years"])
    
    logger.info(f"市场时期切分边界:")
    logger.info(f"  最近期边界: {recent_cutoff.strftime('%Y-%m-%d')}")
    logger.info(f"  中期边界: {middle_cutoff.strftime('%Y-%m-%d')}")
    
    # 按时期分组
    period_samples = {"recent": [], "middle": [], "distant": []}
    
    for sample in all_samples:
        sample_date = sample['date'].to_pydatetime() if hasattr(sample['date'], 'to_pydatetime') else sample['date']
        
        if sample_date >= recent_cutoff:
            period_samples["recent"].append(sample)
        elif sample_date >= middle_cutoff:
            period_samples["middle"].append(sample)
        else:
            period_samples["distant"].append(sample)
    
    # 检查每个时期的样本数量
    min_samples = period_config["min_samples_per_period"]
    valid_periods = {}
    
    for period_name, samples in period_samples.items():
        if len(samples) >= min_samples:
            valid_periods[period_name] = samples
            logger.info(f"  {period_name}期: {len(samples)} 样本 [有效]")
        else:
            logger.warning(f"  {period_name}期: {len(samples)} 样本 (< {min_samples}，跳过)")
    
    if not valid_periods:
        logger.warning("所有时期样本数量不足，使用全部数据")
        return {"all": all_samples}
    
    return valid_periods

def create_3d_datasets_with_distribution(all_samples, train_ratio=0.8, val_ratio=0.1, **kwargs):
    """
    创建3D训练、验证和测试数据集（支持相对化指标和市场时期切分）
    
    Args:
        all_samples (list): 所有样本
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        **kwargs: 传递给Dataset的其他参数
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, stock_distributions)
    """
    # 按市场时期切分样本
    period_samples_dict = split_samples_by_market_periods(all_samples)
    
    # 收集所有时期的训练/验证/测试样本
    all_train_samples = []
    all_val_samples = []
    all_test_samples = []
    
    logger.info(f"开始按时期切分训练/验证/测试集:")
    
    for period_name, period_samples in period_samples_dict.items():
        period_size = len(period_samples)
        train_size = int(period_size * train_ratio)
        val_size = int(period_size * val_ratio)
        
        # 在每个时期内按时间顺序分割（不打乱）
        train_samples = period_samples[:train_size]
        val_samples = period_samples[train_size:train_size + val_size]
        test_samples = period_samples[train_size + val_size:]
        
        # 合并到总样本中
        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_test_samples.extend(test_samples)
        
        logger.info(f"  {period_name}期 ({period_size}样本): 训练{len(train_samples)} + 验证{len(val_samples)} + 测试{len(test_samples)}")
    
    logger.info(f"总数据集:")
    logger.info(f"  训练集: {len(all_train_samples)} 样本")
    logger.info(f"  验证集: {len(all_val_samples)} 样本")
    logger.info(f"  测试集: {len(all_test_samples)} 样本")
    
    # 使用所有训练集构建股票分布（避免数据泄漏）
    logger.info("使用训练集构建股票分布...")
    temp_generator = ThreeDimensionalLabelGenerator(
        look_forward_days=kwargs.get('look_forward_days', 20),
        temperature=kwargs.get('temperature', 0.1),  # 确保使用正确的温度参数
        use_relative_metrics=kwargs.get('use_relative_metrics', True)
    )
    
    # 按股票代码分组训练样本
    stock_samples_dict = {}
    for sample in all_train_samples:
        stock_code = sample.get('stock_code', 'UNKNOWN')
        if stock_code not in stock_samples_dict:
            stock_samples_dict[stock_code] = []
        stock_samples_dict[stock_code].append(sample)
    
    # 为每只股票建立分布
    temp_generator.fit_stock_distributions(stock_samples_dict)
    stock_distributions = temp_generator.relative_calculator.stock_distributions
    
    logger.info(f"股票分布构建完成: {len(stock_distributions)} 只股票")
    
    # 创建数据集，共享相同的分布信息
    dataset_kwargs = kwargs.copy()
    dataset_kwargs['stock_distributions'] = stock_distributions
    
    train_dataset = Market3DClassificationDataset(all_train_samples, **dataset_kwargs)
    val_dataset = Market3DClassificationDataset(all_val_samples, **dataset_kwargs)
    test_dataset = Market3DClassificationDataset(all_test_samples, **dataset_kwargs)
    
    return train_dataset, val_dataset, test_dataset, stock_distributions

# 保持向后兼容
def create_3d_datasets(all_samples, train_ratio=0.8, val_ratio=0.1, centers_config=None, **kwargs):
    """
    创建3D训练、验证和测试数据集（向后兼容版本）
    """
    # 如果没有centers_config，使用新的相对化方法
    if centers_config is None:
        kwargs['use_relative_metrics'] = True
        return create_3d_datasets_with_distribution(all_samples, train_ratio, val_ratio, **kwargs)[:3]
    else:
        # 使用旧的绝对指标方法（向后兼容）
        kwargs['use_relative_metrics'] = False
        kwargs['centers_config'] = centers_config
        return create_3d_datasets_with_distribution(all_samples, train_ratio, val_ratio, **kwargs)[:3]

def test_3d_dataset():
    """
    测试3D数据集
    """
    logger.info("=== 测试3D数据集 ===")
    
    # 创建模拟样本
    import pandas as pd
    np.random.seed(42)
    
    samples = []
    for i in range(10):
        # 模拟价格序列
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 25))
        
        sample = {
            'daily': np.random.randn(60, 12).astype(np.float32),
            'weekly': np.random.randn(52, 12).astype(np.float32),
            'monthly': np.random.randn(24, 12).astype(np.float32),
            'future_prices': prices,
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'stock_code': f'TEST{i%3:03d}'  # 3只测试股票
        }
        samples.append(sample)
    
    # 创建数据集
    dataset = Market3DClassificationDataset(
        samples, 
        look_forward_days=20, 
        temperature=0.002,
        use_relative_metrics=True
    )
    
    # 测试数据加载
    logger.info(f"\n数据集大小: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    logger.info(f"\n第一个样本:")
    logger.info(f"  Daily shape: {sample['daily'].shape}")
    logger.info(f"  Weekly shape: {sample['weekly'].shape}")
    logger.info(f"  Monthly shape: {sample['monthly'].shape}")
    logger.info(f"  Labels 3D keys: {list(sample['labels_3d'].keys())}")
    
    for key, label in sample['labels_3d'].items():
        logger.info(f"  {key} label: {label.numpy()}")
        logger.info(f"  {key} sum: {label.sum().item():.6f}")
    
    # 获取统计信息
    stats = dataset.get_label_statistics()
    logger.info(f"\n标签统计信息:")
    for dim, stat in stats.items():
        logger.info(f"  {dim}:")
        logger.info(f"    平均熵: {stat['mean_entropy']:.4f}")
        logger.info(f"    平均置信度: {stat['mean_confidence']:.4f}")
    
    logger.info("3D数据集测试通过")

if __name__ == '__main__':
    test_3d_dataset()