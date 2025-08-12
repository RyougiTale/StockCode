import sys
import os
import torch
import numpy as np

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

import config
import logger_config

# 初始化日志
logger_config.setup_logging(log_level=config.LOGGING_LEVEL)
logger = logger_config.get_logger(__name__)

def run_mini_pretraining():
    """运行小规模预训练测试"""
    logger.info("=== 小规模预训练测试 ===")
    
    try:
        # 临时修改配置为更小的测试参数
        original_epochs = config.PRETRAINING_EPOCHS
        original_batch_size = config.PRETRAINING_BATCH_SIZE
        original_training_years = config.TRAINING_YEARS
        
        config.PRETRAINING_EPOCHS = 2  # 只训练2个epoch
        config.PRETRAINING_BATCH_SIZE = 8  # 小批次
        config.TRAINING_YEARS = 2  # 只用最近2年数据
        
        # 只用2只股票测试
        test_stocks = config.STOCK_CODES[:2]
        logger.info(f"测试股票: {test_stocks}")
        
        # 1. 数据加载测试
        logger.info("开始数据加载...")
        from data_utils import get_all_samples
        
        all_samples, scalers = get_all_samples(test_stocks)
        logger.info(f"加载样本数: {len(all_samples)}")
        
        if len(all_samples) < 100:
            logger.error(f"样本数太少: {len(all_samples)}")
            return False
        
        # 只使用前200个样本进行测试
        test_samples = all_samples[:200]
        logger.info(f"使用测试样本数: {len(test_samples)}")
        
        # 2. 数据集创建测试
        logger.info("创建数据集...")
        from dataset_3d import create_3d_datasets_with_distribution
        
        train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
            test_samples,
            train_ratio=0.6,
            val_ratio=0.2,
            look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
            temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
            use_relative_metrics=True
        )
        
        logger.info(f"数据集创建成功:")
        logger.info(f"  训练集: {len(train_dataset)} 样本")
        logger.info(f"  验证集: {len(val_dataset)} 样本")
        logger.info(f"  测试集: {len(test_dataset)} 样本")
        logger.info(f"  股票分布: {len(stock_distributions)} 只股票")
        
        # 3. 数据加载器测试
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(train_dataset, batch_size=config.PRETRAINING_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.PRETRAINING_BATCH_SIZE, shuffle=False)
        
        logger.info(f"数据加载器:")
        logger.info(f"  训练批次数: {len(train_loader)}")
        logger.info(f"  验证批次数: {len(val_loader)}")
        
        # 4. 模型创建测试
        logger.info("创建模型...")
        from model_3d import create_3d_model
        
        model = create_3d_model(config).to(config.DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数数量: {total_params:,}")
        
        # 5. 损失函数和优化器
        from model_3d import Multi3DLoss
        
        criterion = Multi3DLoss(weights={
            'return': 1.0,
            'sharpe': 0.8, 
            'drawdown': 0.6
        })
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.PRETRAINING_LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        logger.info("损失函数和优化器创建完成")
        
        # 6. 小规模训练测试
        logger.info("开始小规模训练测试...")
        from engine_3d import train_one_epoch_3d, evaluate_3d, format_3d_results
        
        for epoch in range(config.PRETRAINING_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{config.PRETRAINING_EPOCHS}")
            
            # 训练
            train_losses, train_accs = train_one_epoch_3d(
                model, train_loader, criterion, optimizer, config.DEVICE,
                grad_clip_norm=config.GRAD_CLIP_NORM
            )
            
            # 验证  
            val_losses, val_accs = evaluate_3d(model, val_loader, criterion, config.DEVICE)
            
            # 格式化输出
            train_loss_str, train_acc_str = format_3d_results(train_losses, train_accs)
            val_loss_str, val_acc_str = format_3d_results(val_losses, val_accs)
            
            logger.info(f"  训练损失: {train_loss_str}")
            logger.info(f"  验证损失: {val_loss_str}")
            
            # 检查损失是否合理
            if torch.isnan(torch.tensor(train_losses['total'])) or train_losses['total'] > 100:
                logger.error("训练出现问题，损失异常")
                return False
        
        # 7. 模型保存测试
        logger.info("测试模型保存...")
        test_model_path = "test_pretraining_model.pth"
        
        try:
            torch.save(model.state_dict(), test_model_path)
            logger.info("模型保存成功")
            
            # 测试加载
            model.load_state_dict(torch.load(test_model_path, map_location=config.DEVICE))
            logger.info("模型加载成功")
            
            # 清理测试文件
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
                
        except Exception as e:
            logger.error(f"模型保存/加载测试失败: {e}")
            return False
        
        # 恢复原始配置
        config.PRETRAINING_EPOCHS = original_epochs
        config.PRETRAINING_BATCH_SIZE = original_batch_size
        config.TRAINING_YEARS = original_training_years
        
        logger.info("小规模预训练测试完全通过!")
        return True
        
    except Exception as e:
        logger.error(f"小规模预训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logger.info("开始完整预训练系统测试")
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if run_mini_pretraining():
        logger.info("🎉 预训练系统测试成功!")
        logger.info("")
        logger.info("系统已经准备就绪，可以进行完整预训练:")
        logger.info("1. 确保配置正确（config.py中的股票列表、epochs等）")
        logger.info("2. 运行完整预训练脚本")
        logger.info("3. 监控训练过程和模型性能")
        logger.info("")
        logger.info("建议的完整预训练步骤:")
        logger.info("- 增加PRETRAINING_EPOCHS到100-1000")
        logger.info("- 使用更多股票（10-50只）进行预训练")
        logger.info("- 设置合适的学习率和批次大小")
        return True
    else:
        logger.error("预训练系统测试失败，请检查问题")
        return False

if __name__ == '__main__':
    main()