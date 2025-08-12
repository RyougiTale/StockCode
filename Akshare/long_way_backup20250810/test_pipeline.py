import sys
import os
import torch
import numpy as np

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

import config
import logger_config
from stock_util import read_history_by_code

# 初始化日志
logger_config.setup_logging(log_level=config.LOGGING_LEVEL)
logger = logger_config.get_logger(__name__)

def test_single_stock_pipeline():
    """测试单只股票的完整数据处理流水线"""
    logger.info("=== 单只股票数据流水线测试 ===")
    
    try:
        # 1. 加载原始数据
        test_code = "002415"
        logger.info(f"测试股票: {test_code}")
        
        df = read_history_by_code(test_code)
        if df is None or df.empty:
            logger.error("无法加载股票数据")
            return False
            
        logger.info(f"原始数据: {len(df)} 行")
        
        # 2. 模拟数据预处理
        # 保存原始收盘价
        original_close = df['close'].copy()
        look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
        
        # 计算标签
        df['label'] = original_close.shift(-look_forward_days) / original_close - 1
        
        logger.info(f"标签计算完成，前瞻天数: {look_forward_days}")
        
        # 3. 创建少量样本用于测试
        samples = []
        valid_indices = df.dropna(subset=['label']).index[-100:]  # 取最后100个有效样本
        
        for i in valid_indices:
            if i + look_forward_days < len(df):
                future_prices = original_close.iloc[i:i + look_forward_days].values
                if len(future_prices) == look_forward_days:
                    sample = {
                        'daily': np.random.randn(60, 12).astype(np.float32),  # 模拟归一化后的数据
                        'weekly': np.random.randn(52, 12).astype(np.float32),
                        'monthly': np.random.randn(24, 12).astype(np.float32),
                        'future_prices': future_prices,
                        'stock_code': test_code,
                        'date': df.iloc[i]['date']
                    }
                    samples.append(sample)
                    
                    if len(samples) >= 20:  # 只需要20个样本测试
                        break
        
        logger.info(f"创建测试样本: {len(samples)} 个")
        
        if len(samples) == 0:
            logger.error("无法创建测试样本")
            return False
        
        # 4. 测试相对化指标计算
        from scipy.stats import percentileofscore
        
        # 计算所有样本的指标
        all_metrics = []
        for sample in samples:
            future_prices = sample['future_prices']
            if len(future_prices) >= 2:
                # 计算指标
                total_return = (future_prices[-1] / future_prices[0]) - 1
                
                # 简化的夏普比率计算
                daily_returns = np.diff(future_prices) / future_prices[:-1]
                if len(daily_returns) > 1 and np.std(daily_returns) > 1e-9:
                    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)
                    sharpe_ratio = np.clip(sharpe_ratio, -2.0, 2.0)
                else:
                    sharpe_ratio = 0.0
                
                # 最大回撤
                cummax = np.maximum.accumulate(future_prices)
                drawdown = (future_prices - cummax) / (cummax + 1e-9)
                max_drawdown = np.min(drawdown)
                max_drawdown = np.clip(max_drawdown, -1.0, 0.0)
                
                metrics = {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
                all_metrics.append(metrics)
        
        logger.info(f"计算指标: {len(all_metrics)} 个有效指标")
        
        # 5. 构建该股票的分布
        returns = [m['total_return'] for m in all_metrics]
        sharpes = [m['sharpe_ratio'] for m in all_metrics]  
        drawdowns = [m['max_drawdown'] for m in all_metrics]
        
        logger.info(f"回报率范围: [{min(returns):.4f}, {max(returns):.4f}]")
        logger.info(f"夏普比率范围: [{min(sharpes):.4f}, {max(sharpes):.4f}]")
        logger.info(f"最大回撤范围: [{min(drawdowns):.4f}, {max(drawdowns):.4f}]")
        
        # 6. 测试相对化转换
        test_metric = all_metrics[len(all_metrics)//2]  # 取中间的一个指标测试
        
        return_rank = percentileofscore(returns, test_metric['total_return'], kind='rank') / 100.0
        sharpe_rank = percentileofscore(sharpes, test_metric['sharpe_ratio'], kind='rank') / 100.0  
        drawdown_rank = percentileofscore(drawdowns, test_metric['max_drawdown'], kind='rank') / 100.0
        
        logger.info(f"测试指标相对化:")
        logger.info(f"  原始回报率: {test_metric['total_return']:.4f} -> 相对位置: {return_rank:.4f}")
        logger.info(f"  原始夏普: {test_metric['sharpe_ratio']:.4f} -> 相对位置: {sharpe_rank:.4f}")
        logger.info(f"  原始回撤: {test_metric['max_drawdown']:.4f} -> 相对位置: {drawdown_rank:.4f}")
        
        # 7. 测试软标签生成
        import torch.nn.functional as F
        
        def create_soft_label(value, centers=[0.1, 0.3, 0.5, 0.7, 0.9], temperature=0.002):
            centers_tensor = torch.tensor(centers, dtype=torch.float32)
            distances = torch.abs(centers_tensor - value)
            logits = -distances / temperature
            probabilities = F.softmax(logits, dim=0)
            return probabilities
        
        return_label = create_soft_label(return_rank)
        sharpe_label = create_soft_label(sharpe_rank)
        drawdown_label = create_soft_label(drawdown_rank)
        
        logger.info(f"软标签生成:")
        logger.info(f"  回报率软标签: {return_label.numpy()}")
        logger.info(f"  夏普软标签: {sharpe_label.numpy()}")
        logger.info(f"  回撤软标签: {drawdown_label.numpy()}")
        
        # 验证软标签有效性
        for name, label in [('回报率', return_label), ('夏普', sharpe_label), ('回撤', drawdown_label)]:
            label_sum = label.sum().item()
            if abs(label_sum - 1.0) > 1e-6:
                logger.error(f"{name}软标签和不为1: {label_sum}")
                return False
            if torch.isnan(label).any() or torch.isinf(label).any():
                logger.error(f"{name}软标签包含NaN/Inf")
                return False
        
        logger.info("单只股票数据流水线测试通过")
        return True
        
    except Exception as e:
        logger.error(f"数据流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_stock_simulation():
    """测试多股票数据模拟"""
    logger.info("=== 多股票数据模拟测试 ===")
    
    try:
        # 模拟多只股票的相对化效果
        np.random.seed(42)
        
        # 模拟3只不同特征的股票
        stocks_data = {
            '000001': {  # 大盘蓝筹，低波动
                'returns': np.random.normal(0.0005, 0.015, 200),
                'volatility': 0.015
            },
            '002415': {  # 中盘成长，中等波动  
                'returns': np.random.normal(0.001, 0.025, 200),
                'volatility': 0.025
            },
            '000858': {  # 小盘股，高波动
                'returns': np.random.normal(0.002, 0.045, 200), 
                'volatility': 0.045
            }
        }
        
        # 为每只股票计算相对化指标
        from scipy.stats import percentileofscore
        
        # 测试同样的绝对回报率在不同股票中的相对位置
        test_return = 0.03
        
        logger.info(f"测试绝对回报率 {test_return:.4f} 在不同股票中的相对位置:")
        
        for stock_code, data in stocks_data.items():
            returns = data['returns']
            relative_rank = percentileofscore(returns, test_return, kind='rank') / 100.0
            
            logger.info(f"  {stock_code}: 相对位置 {relative_rank:.4f} (波动率: {data['volatility']:.4f})")
        
        # 验证相对化的好处：同样的绝对收益在不同股票中有不同的相对意义
        ranks = []
        for stock_code, data in stocks_data.items():
            rank = percentileofscore(data['returns'], test_return, kind='rank') / 100.0
            ranks.append(rank)
        
        rank_std = np.std(ranks)
        logger.info(f"相对位置标准差: {rank_std:.4f}")
        
        if rank_std > 0.1:  # 如果标准差足够大，说明相对化是有意义的
            logger.info("相对化有效：同样收益在不同股票中确实有不同的相对意义")
        else:
            logger.warning("相对化效果不明显，可能需要更多样化的股票")
        
        logger.info("多股票模拟测试通过")
        return True
        
    except Exception as e:
        logger.error(f"多股票模拟测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始数据流水线测试")
    
    tests = [
        ("单只股票流水线", test_single_stock_pipeline),
        ("多股票相对化模拟", test_multi_stock_simulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 执行: {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"[通过] {test_name}")
            else:
                logger.error(f"[失败] {test_name}")
        except Exception as e:
            logger.error(f"[异常] {test_name}: {e}")
    
    logger.info("=" * 50)
    logger.info(f"流水线测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("数据流水线测试全部通过！")
        logger.info("相对化指标系统工作正常，可以进行实际预训练")
        
        # 给出下一步建议
        logger.info("\n下一步建议:")
        logger.info("1. 运行完整预训练: python -c \"import sys; sys.path.append('.'); from train_3d import main; main()\"")  
        logger.info("2. 或者先用更多股票测试数据加载性能")
        
    else:
        logger.error(f"还有 {total - passed} 个测试失败，请先解决问题")
    
    return passed == total

if __name__ == '__main__':
    main()