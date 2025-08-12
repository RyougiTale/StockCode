import sys
import os
import torch

# 确保可以导入模块
sys.path.append(os.path.dirname(__file__))

# 导入配置和日志
from config import *
from logger_config import setup_logging, get_logger

# 初始化日志
setup_logging(log_level=LOGGING_LEVEL)
logger = get_logger(__name__)

def test_basic_imports():
    """测试基本导入功能"""
    logger.info("=== 测试基本导入 ===")
    
    try:
        from data_utils import get_all_samples, get_stock_codes_for_training
        logger.info("✓ data_utils 导入成功")
        
        from label_3d_generator import get_label_generator
        logger.info("✓ label_3d_generator 导入成功")
        
        from dataset_3d import create_3d_datasets_with_distribution
        logger.info("✓ dataset_3d 导入成功")
        
        from model_3d import create_3d_model
        logger.info("✓ model_3d 导入成功")
        
        logger.info("所有基本导入测试通过")
        return True
        
    except Exception as e:
        logger.error(f"导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stock_code_loading():
    """测试股票代码加载"""
    logger.info("=== 测试股票代码加载 ===")
    
    try:
        from data_utils import get_stock_codes_for_training
        
        stock_codes = get_stock_codes_for_training()
        logger.info(f"获取到股票代码: {stock_codes}")
        logger.info(f"股票数量: {len(stock_codes)}")
        
        if stock_codes:
            logger.info("✓ 股票代码加载测试通过")
            return True
        else:
            logger.error("✗ 未获取到任何股票代码")
            return False
            
    except Exception as e:
        logger.error(f"股票代码加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_data_loading():
    """测试样本数据加载（只加载少量数据）"""
    logger.info("=== 测试样本数据加载 ===")
    
    try:
        from data_utils import get_all_samples
        
        # 只测试第一只股票
        test_codes = STOCK_CODES[:1]
        logger.info(f"测试股票: {test_codes}")
        
        all_samples, scalers = get_all_samples(test_codes)
        logger.info(f"获取样本数: {len(all_samples)}")
        
        if all_samples:
            sample = all_samples[0]
            logger.info(f"样本键: {list(sample.keys())}")
            logger.info(f"样本股票代码: {sample.get('stock_code', 'N/A')}")
            logger.info(f"日线数据形状: {sample['daily'].shape}")
            logger.info(f"周线数据形状: {sample['weekly'].shape}")
            logger.info(f"月线数据形状: {sample['monthly'].shape}")
            logger.info(f"未来价格数量: {len(sample['future_prices'])}")
            
            logger.info("✓ 样本数据加载测试通过")
            return True, all_samples[:10], scalers  # 只返回前10个样本用于后续测试
        else:
            logger.error("✗ 未获取到任何样本数据")
            return False, [], {}
            
    except Exception as e:
        logger.error(f"样本数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, [], {}

def test_relative_metrics():
    """测试相对化指标功能"""
    logger.info("=== 测试相对化指标 ===")
    
    try:
        from label_3d_generator import test_relative_metrics
        test_relative_metrics()
        logger.info("✓ 相对化指标测试通过")
        return True
        
    except Exception as e:
        logger.error(f"相对化指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation(test_samples):
    """测试数据集创建"""
    logger.info("=== 测试数据集创建 ===")
    
    try:
        from dataset_3d import create_3d_datasets_with_distribution
        
        if not test_samples:
            logger.error("没有测试样本")
            return False
            
        train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
            test_samples,
            train_ratio=0.6,
            val_ratio=0.2,
            look_forward_days=SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        logger.info(f"测试集大小: {len(test_dataset)}")
        logger.info(f"股票分布数量: {len(stock_distributions)}")
        
        # 测试单个样本
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"单个样本键: {list(sample.keys())}")
            logger.info(f"3D标签键: {list(sample['labels_3d'].keys())}")
            
            for key, label in sample['labels_3d'].items():
                logger.info(f"{key}标签: {label.numpy()}")
                logger.info(f"{key}标签和: {label.sum().item():.6f}")
        
        logger.info("✓ 数据集创建测试通过")
        return True
        
    except Exception as e:
        logger.error(f"数据集创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    logger.info("=== 测试模型创建 ===")
    
    try:
        from model_3d import create_3d_model
        
        model = create_3d_model(sys.modules[__name__])  # 传递当前模块作为config
        logger.info(f"模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        logger.info("✓ 模型创建测试通过")
        return True
        
    except Exception as e:
        logger.error(f"模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始预训练系统测试")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"训练阶段: {TRAINING_PHASE}")
    logger.info(f"预设股票: {STOCK_CODES}")
    
    # 依次进行各项测试
    tests_passed = 0
    total_tests = 6
    
    # 1. 基本导入测试
    if test_basic_imports():
        tests_passed += 1
    
    # 2. 股票代码加载测试  
    if test_stock_code_loading():
        tests_passed += 1
    
    # 3. 样本数据加载测试
    success, test_samples, scalers = test_sample_data_loading()
    if success:
        tests_passed += 1
    
    # 4. 相对化指标测试
    if test_relative_metrics():
        tests_passed += 1
    
    # 5. 数据集创建测试
    if test_samples and test_dataset_creation(test_samples):
        tests_passed += 1
    
    # 6. 模型创建测试
    if test_model_creation():
        tests_passed += 1
    
    # 结果汇总
    logger.info("=" * 50)
    logger.info(f"测试结果: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        logger.info("🎉 所有测试通过！预训练系统准备就绪")
        logger.info("可以运行完整预训练: python -m long_way.train_3d")
    else:
        logger.error(f"❌ {total_tests - tests_passed} 个测试失败，请检查问题")
    
    return tests_passed == total_tests

if __name__ == '__main__':
    main()