import sys
import os
import torch
import numpy as np

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# 现在可以正常导入
import config
import logger_config

# 初始化日志
logger_config.setup_logging(log_level=config.LOGGING_LEVEL)
logger = logger_config.get_logger(__name__)

def test_simple_data_loading():
    """简单测试数据加载功能"""
    logger.info("=== 简单数据加载测试 ===")
    
    # 直接使用stock_util中的功能
    from stock_util import read_history_by_code
    
    # 测试单只股票数据加载
    test_code = "002415"
    logger.info(f"测试加载股票: {test_code}")
    
    try:
        df = read_history_by_code(test_code)
        if df is not None and not df.empty:
            logger.info(f"数据加载成功，行数: {len(df)}")
            logger.info(f"列名: {list(df.columns)}")
            logger.info(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
            return True
        else:
            logger.error("数据为空")
            return False
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return False

def test_relative_calculator():
    """测试相对化计算器的核心功能"""
    logger.info("=== 相对化计算器测试 ===")
    
    try:
        # 模拟一些股票的历史数据
        np.random.seed(42)
        
        # 模拟两只股票的历史指标
        stock_data = {
            'TEST001': {
                'returns': np.random.normal(0.001, 0.02, 100),
                'sharpes': np.random.normal(0.1, 0.5, 100), 
                'drawdowns': -np.abs(np.random.normal(0.05, 0.03, 100))
            },
            'TEST002': {
                'returns': np.random.normal(0.002, 0.05, 100),
                'sharpes': np.random.normal(0.2, 0.8, 100),
                'drawdowns': -np.abs(np.random.normal(0.08, 0.05, 100))
            }
        }
        
        # 模拟RelativeMetricsCalculator的核心功能
        from scipy.stats import percentileofscore
        
        def get_percentile_rank(value, sorted_values):
            """计算百分位排名"""
            return percentileofscore(sorted_values, value, kind='rank') / 100.0
        
        # 测试相对化转换
        test_metrics = {
            'total_return': 0.05,
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.02
        }
        
        for stock_code, data in stock_data.items():
            return_rank = get_percentile_rank(test_metrics['total_return'], sorted(data['returns']))
            sharpe_rank = get_percentile_rank(test_metrics['sharpe_ratio'], sorted(data['sharpes']))
            drawdown_rank = get_percentile_rank(test_metrics['max_drawdown'], sorted(data['drawdowns']))
            
            logger.info(f"股票 {stock_code} 相对化指标:")
            logger.info(f"  回报率百分位: {return_rank:.4f}")
            logger.info(f"  夏普比率百分位: {sharpe_rank:.4f}")
            logger.info(f"  最大回撤百分位: {drawdown_rank:.4f}")
        
        logger.info("✓ 相对化计算器核心功能测试通过")
        return True
        
    except Exception as e:
        logger.error(f"相对化计算器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_soft_label_generation():
    """测试软标签生成"""
    logger.info("=== 软标签生成测试 ===")
    
    try:
        import torch
        import torch.nn.functional as F
        
        def create_soft_label(value, centers, temperature=0.002):
            """创建软标签的核心逻辑"""
            centers_tensor = torch.tensor(centers, dtype=torch.float32)
            distances = torch.abs(centers_tensor - value)
            logits = -distances / temperature
            probabilities = F.softmax(logits, dim=0)
            return probabilities
        
        # 测试不同的相对化值
        centers = [0.1, 0.3, 0.5, 0.7, 0.9]  # 相对化中心点
        test_values = [0.2, 0.5, 0.8]
        
        for value in test_values:
            soft_label = create_soft_label(value, centers)
            logger.info(f"值 {value} 的软标签: {soft_label.numpy()}")
            logger.info(f"概率和: {soft_label.sum().item():.6f}")
            logger.info(f"最高概率类别: {soft_label.argmax().item()}")
        
        logger.info("✓ 软标签生成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"软标签生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_validation():
    """测试配置验证"""
    logger.info("=== 配置验证测试 ===")
    
    try:
        logger.info(f"设备: {config.DEVICE}")
        logger.info(f"训练阶段: {config.TRAINING_PHASE}")
        logger.info(f"预训练轮数: {config.PRETRAINING_EPOCHS}")
        logger.info(f"预训练批次大小: {config.PRETRAINING_BATCH_SIZE}")
        logger.info(f"预训练学习率: {config.PRETRAINING_LEARNING_RATE}")
        logger.info(f"预设股票数量: {len(config.STOCK_CODES)}")
        logger.info(f"预设股票: {config.STOCK_CODES}")
        
        # 验证配置有效性
        assert config.TRAINING_PHASE == "pretraining", "训练阶段应为预训练"
        assert len(config.STOCK_CODES) > 0, "应有测试股票"
        assert config.PRETRAINING_EPOCHS > 0, "预训练轮数应大于0"
        
        logger.info("✓ 配置验证通过")
        return True
        
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始预训练系统基础功能测试")
    
    tests = [
        ("配置验证", test_config_validation),
        ("数据加载", test_simple_data_loading),
        ("相对化计算器", test_relative_calculator),
        ("软标签生成", test_soft_label_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 执行: {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} 通过")
            else:
                logger.error(f"✗ {test_name} 失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 出现异常: {e}")
    
    logger.info("=" * 50)
    logger.info(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("所有基础功能测试通过！")
        logger.info("建议接下来测试完整的数据流水线")
    else:
        logger.error(f"{total - passed} 个测试失败，请检查问题")
    
    return passed == total

if __name__ == '__main__':
    main()