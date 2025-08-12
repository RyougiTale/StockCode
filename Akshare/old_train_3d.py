import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

# 导入我们的3D模块
from . import config
from .model_3d import MultiOutput3DModel, Multi3DLoss, create_3d_model
from .dataset_3d import Market3DClassificationDataset, create_3d_datasets_with_distribution
from .engine_3d import train_one_epoch_3d, evaluate_3d, format_3d_results
from .data_utils import get_all_samples
from .label_3d_generator import ThreeDimensionalLabelGenerator
from .logger_config import get_logger, log_performance, setup_logging

# 初始化日志系统
setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

@log_performance("全局指标分位数计算")
def calculate_global_centers(all_samples, look_forward_days):
    """
    遍历所有样本，计算全局的指标分位数作为类别中心。
    """
    logger.info("开始计算全局指标分位数...")
    temp_label_generator = ThreeDimensionalLabelGenerator(look_forward_days=look_forward_days)
    all_metrics = []

    for sample in tqdm(all_samples, desc="分析全局指标分布"):
        if 'future_prices' in sample and len(sample['future_prices']) > 1:
            future_prices = pd.Series(sample['future_prices'])
            metrics = temp_label_generator.calculate_future_metrics(future_prices)
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        logger.warning("无法计算全局指标，将使用默认中心点")
        return None

    df_metrics = pd.DataFrame(all_metrics)
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    global_centers = {}
    center_dims = {
        'return': 'total_return',
        'sharpe': 'sharpe_ratio',
        'drawdown': 'max_drawdown'
    }

    if config.DEBUG_MODE:
        logger.debug("全局指标分位数结果:")
    for center_key, metric_key in center_dims.items():
        percentiles = df_metrics[metric_key].quantile(quantiles).values
        
        # 后处理：确保中心点唯一，避免重复
        unique_percentiles = np.sort(np.unique(percentiles))
        if len(unique_percentiles) < len(percentiles):
            logger.warning(f"{center_key.upper()}维度出现重复中心点，将添加微小扰动")
            for i in range(1, len(percentiles)):
                if percentiles[i] <= percentiles[i-1]:
                    percentiles[i] = percentiles[i-1] + 1e-5 # 添加一个微小的偏移量
        
        global_centers[center_key] = percentiles
        if config.DEBUG_MODE:
            logger.debug(f"  - {center_key.upper()}中心点: {np.round(percentiles, 4)}")
        else:
            logger.info(f"{center_key.upper()}中心点计算完成")

    return global_centers

@log_performance("3D软标签模型训练")
def main():
    """3D软标签模型训练主函数"""
    logger.info("=== 3D软标签模型训练开始 ===")
    
    # --- 1. 准备数据 ---
    logger.info("开始数据准备...")
    all_samples, scalers = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        logger.error("无法获取样本数据")
        return
    
    logger.info(f"总样本数: {len(all_samples)}")
    
    # --- 1b. 时间筛选 ---
    if config.TRAINING_YEARS is not None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * config.TRAINING_YEARS)
        
        original_count = len(all_samples)
        all_samples = [s for s in all_samples if s['date'].to_pydatetime() >= start_date]
        logger.info(f"筛选最近 {config.TRAINING_YEARS} 年数据: {len(all_samples)}/{original_count} 样本")
    
    # --- 1c. 计算全局类别中心 ---
    global_centers = calculate_global_centers(
        all_samples,
        config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    )

    # --- 2. 创建3D数据集 ---
    logger.info("开始创建3D数据集...")
    
    # 使用新的相对化数据集创建方法
    train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
        all_samples,
        train_ratio=0.8,
        val_ratio=0.1,
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True  # 启用相对化指标
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    logger.info("3D数据加载器创建完成:")
    if config.DEBUG_MODE:
        logger.debug(f"训练批次: {len(train_loader)}")
        logger.debug(f"验证批次: {len(val_loader)}")
        logger.debug(f"测试批次: {len(test_loader)}")
    else:
        logger.info(f"数据加载器: 训练={len(train_loader)}, 验证={len(val_loader)}, 测试={len(test_loader)}")
    
    # --- 3. 创建3D模型 ---
    logger.info("创建3D模型...")
    model = create_3d_model(config).to(config.DEVICE)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # --- 4. 创建损失函数和优化器 ---
    logger.info("初始化损失函数和优化器...")
    
    # 3D多任务损失函数
    criterion = Multi3DLoss(weights={
        'return': 1.0,    # 回报率权重最高
        'sharpe': 0.8,    # 夏普比率次之
        'drawdown': 0.6   # 最大回撤权重较低
    })
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    logger.info("损失函数和优化器初始化完成")
    
    # --- 5. 训练循环 ---
    logger.info(f"开始训练 ({config.DEVICE}, {config.EPOCHS} epochs)...")
    
    # 创建模型保存目录
    model_3d_dir = os.path.join(config.MODEL_DIR, "3d_models")
    os.makedirs(model_3d_dir, exist_ok=True)
    
    # 最佳模型跟踪（参考原版train.py的逻辑）
    best_models = {
        'total_loss': [(float('inf'), '')] * 3,  # 保存top3损失模型
        'return_acc@1': [(0.0, '')],             # 只保存top1准确率模型
        'sharpe_acc@1': [(0.0, '')],
        'drawdown_acc@1': [(0.0, '')]
    }
    
    for epoch in range(config.EPOCHS):
        if config.DEBUG_MODE:
            logger.debug(f"=== Epoch {epoch+1}/{config.EPOCHS} ===")
        
        # 训练
        train_losses, train_accs = train_one_epoch_3d(
            model, train_loader, criterion, optimizer, config.DEVICE,
            grad_clip_norm=getattr(config, 'GRAD_CLIP_NORM', 1.0)
        )
        
        # 验证
        val_losses, val_accs = evaluate_3d(model, val_loader, criterion, config.DEVICE)
        
        # 格式化输出
        train_loss_str, train_acc_str = format_3d_results(train_losses, train_accs)
        val_loss_str, val_acc_str = format_3d_results(val_losses, val_accs)
        
        logger.info(f"Epoch {epoch+1:3d}/{config.EPOCHS} | 训练损失: {train_loss_str} | 验证损失: {val_loss_str}")
        if config.DEBUG_MODE:
            logger.debug(f"  训练准确率: {train_acc_str}")
            logger.debug(f"  验证准确率: {val_acc_str}")
        
        # --- 保存最佳模型 ---
        save_best_3d_models(model, val_losses, val_accs, best_models, model_3d_dir, epoch + 1)
        
        # 早停检查（可选）
        if epoch > 50 and val_losses['total'] > 10.0:  # 如果损失过大，可能有问题
            logger.warning("验证损失过大，可能存在训练问题")
    
    logger.info("训练完成")
    
    # --- 7. 最终测试 ---
    logger.info("开始最终测试评估...")
    # 选择损失最小的模型进行测试
    best_loss_models = sorted(best_models['total_loss'], key=lambda x: x[0])
    best_model_path = best_loss_models[0][1]
    
    if os.path.exists(best_model_path):
        logger.info(f"加载最佳模型: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        
        test_losses, test_accs = evaluate_3d(model, test_loader, criterion, config.DEVICE)
        test_loss_str, test_acc_str = format_3d_results(test_losses, test_accs)
        
        logger.info(f"最终测试结果 - 损失: {test_loss_str}")
        if config.DEBUG_MODE:
            logger.debug(f"  测试准确率: {test_acc_str}")
    else:
        logger.error("未找到最佳模型文件")
    
    logger.info("=== 3D软标签模型训练完成 ===")

def save_best_3d_models(model, val_losses, val_accs, best_models, save_dir, epoch):
    """
    保存3D模型的最佳版本 (loss top3, acc top1 per dim)，文件名固定排名。
    """
    min_epoch_to_save = 5
    min_acc_threshold = 0.15

    if epoch < min_epoch_to_save:
        if config.DEBUG_MODE:
            logger.debug(f"Epoch {epoch}: 训练初期，暂不保存模型")
        return

    # --- 1. 保存最佳总损失模型 (Top 3) ---
    current_loss = val_losses['total']
    current_best_losses = [v[0] for v in best_models['total_loss']]
    
    if current_loss < max(current_best_losses):
        # 找到要替换的位置（损失最大的那个）
        idx_to_replace = np.argmax(current_best_losses)
        old_path = best_models['total_loss'][idx_to_replace][1]
        
        # 新文件名使用固定的槽位号
        new_path = os.path.join(save_dir, f"best_loss_top_{idx_to_replace+1}.pth")
        
        # 删除旧文件并保存新文件
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
        torch.save(model.state_dict(), new_path)
        
        # 更新记录
        best_models['total_loss'][idx_to_replace] = (current_loss, new_path)
        logger.info(f"  更新Top-{idx_to_replace+1}损失模型: {os.path.basename(new_path)} (损失: {max(current_best_losses):.6f} -> {current_loss:.6f})")

    # --- 2. 保存各维度最佳准确率模型 (Top 1) ---
    for dim in ['return', 'sharpe', 'drawdown']:
        key = f'{dim}_acc@1'
        if key in best_models and dim in val_accs:
            current_acc = val_accs[dim]['acc@1']
            best_acc = best_models[key][0][0]
            
            if current_acc > best_acc and current_acc > min_acc_threshold:
                old_path = best_models[key][0][1]
                
                # 新文件名是固定的
                new_path = os.path.join(save_dir, f"best_{dim}_acc.pth")

                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                torch.save(model.state_dict(), new_path)
                
                best_models[key][0] = (current_acc, new_path)
                logger.info(f"  更新最佳{dim}准确率模型: {os.path.basename(new_path)} (准确率: {best_acc:.4f} -> {current_acc:.4f})")

def analyze_3d_training_data():
    """
    分析3D训练数据的特征
    """
    logger.info("=== 3D训练数据分析 ===")
    
    # 获取数据
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        logger.error("无法获取样本数据")
        return
    
    # 创建数据集进行分析
    dataset = Market3DClassificationDataset(
        all_samples[:1000],  # 只分析前1000个样本
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"]
    )
    
    # 获取统计信息
    stats = dataset.get_label_statistics()
    
    logger.info("开始3D标签统计分析:")
    if config.DEBUG_MODE:
        for dim, stat in stats.items():
            logger.debug(f"{dim.upper()}:")
            logger.debug(f"  平均熵: {stat['mean_entropy']:.4f} (不确定性)")
            logger.debug(f"  熵标准差: {stat['std_entropy']:.4f}")
            logger.debug(f"  平均置信度: {stat['mean_confidence']:.4f}")
            logger.debug(f"  置信度标准差: {stat['std_confidence']:.4f}")
    else:
        avg_entropy = np.mean([stat['mean_entropy'] for stat in stats.values()])
        avg_confidence = np.mean([stat['mean_confidence'] for stat in stats.values()])
        logger.info(f"平均熵: {avg_entropy:.4f}, 平均置信度: {avg_confidence:.4f}")
    
    logger.info("分析完成")

def test_3d_training_pipeline():
    """
    测试3D训练管道
    """
    logger.info("=== 测试3D训练管道 ===")
    
    try:
        # 测试数据分析
        analyze_3d_training_data()
        
        logger.info("3D训练管道测试通过")
        logger.info("可以运行完整训练: python -m long_way.train_3d")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        if config.DEBUG_MODE:
            traceback.print_exc()

if __name__ == '__main__':
    # 可以选择运行完整训练或仅测试
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_3d_training_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze_3d_training_data()
    else:
        main()