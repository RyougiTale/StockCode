import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import joblib
import heapq
from datetime import datetime, timedelta

# 导入我们自己的模块
from . import config
from .model import MultiEncoderFusionModel
from .data_utils import get_all_samples
from .dataset import MarketClassificationDataset
from .engine import train_one_epoch, evaluate
from .logger_config import get_logger, log_performance, setup_logging

# 初始化日志系统
setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

@log_performance("训练主流程")
def main():
    # --- 1. 准备数据 ---
    logger.info("开始数据准备...")
    all_samples, scalers = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        logger.error("未创建任何数据样本，退出程序")
        return
    
    logger.info(f"共创建 {len(all_samples)} 个样本")

    # --- 1b. 根据配置筛选最近N年的样本 ---
    if config.TRAINING_YEARS is not None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * config.TRAINING_YEARS)
        
        original_count = len(all_samples)
        all_samples = [s for s in all_samples if s['date'].to_pydatetime() >= start_date]
        logger.info(f"筛选最近 {config.TRAINING_YEARS} 年的数据: 保留 {len(all_samples)}/{original_count} 个样本")
    
    # --- 2. 创建数据集和数据加载器 ---
    full_dataset = MarketClassificationDataset(all_samples)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = int(0.05 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    indices = list(range(len(full_dataset)))
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(full_dataset, indices[train_size : train_size + val_size])
    test_dataset = torch.utils.data.Subset(full_dataset, indices[train_size + val_size :])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    logger.info(f"数据集划分 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

    # 打印数据集的切分日期
    if config.DEBUG_MODE:
        train_end_date = all_samples[indices[train_size - 1]]['date']
        val_start_date = all_samples[indices[train_size]]['date']
        val_end_date = all_samples[indices[train_size + val_size - 1]]['date']
        test_start_date = all_samples[indices[train_size + val_size]]['date']
        
        logger.debug(f"训练集结束日期: {train_end_date.strftime('%Y-%m-%d')}")
        logger.debug(f"验证集日期范围: {val_start_date.strftime('%Y-%m-%d')} 至 {val_end_date.strftime('%Y-%m-%d')}")
        logger.debug(f"测试集开始日期: {test_start_date.strftime('%Y-%m-%d')}")
    logger.info("数据准备完成")

    # --- 2c. 打印标签分布以检查平衡性 ---
    if config.DEBUG_MODE or config.ENABLE_DATA_VALIDATION:
        logger.debug("分析标签分布...")
        labels = [s['label'] for s in all_samples]
        
        # 找到每个回报率最接近的中心点，作为其“硬标签”
        class_centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
        hard_labels = [np.argmin(np.abs(class_centers - l)) for l in labels]
        
        class_counts = np.bincount(hard_labels, minlength=config.NUM_CLASSES)
        
        logger.debug("类别分布（基于最近中心点）:")
        for i, count in enumerate(class_counts):
            logger.debug(f"  - 中心 {class_centers[i]*100: >5.1f}%: {count} 个样本")
    
    # --- 2d. 检查标签分布的数值特征 ---
    if config.DEBUG_MODE or config.ENABLE_DATA_VALIDATION:
        logger.debug("分析标签统计...")
        labels_array = np.array(labels)
        logger.debug(f"标签统计: 数量={len(labels_array)}, 均值={labels_array.mean():.6f}, 标准差={labels_array.std():.6f}")
        logger.debug(f"最小值={labels_array.min():.6f}, 最大值={labels_array.max():.6f}, 中位数={np.median(labels_array):.6f}")
        
        # 检查极端值
        extreme_threshold = 0.3  # 30%的回报率
        extreme_labels = labels_array[np.abs(labels_array) > extreme_threshold]
        if len(extreme_labels) > 0:
            logger.warning(f"极端标签 (>{extreme_threshold*100}%): {len(extreme_labels)} 个 ({len(extreme_labels)/len(labels_array)*100:.1f}%)")
            logger.warning(f"极端值范围: [{extreme_labels.min():.6f}, {extreme_labels.max():.6f}]")
            
        # 检查是否有无穷大或NaN
        nan_count = np.isnan(labels_array).sum()
        inf_count = np.isinf(labels_array).sum()
        if nan_count > 0 or inf_count > 0:
            logger.error(f"发现无效标签: NaN={nan_count}, Inf={inf_count}")

    # --- 3. 初始化模型、损失函数和优化器 ---
    daily_config = {'feature_size': len(config.FEATURE_COLUMNS['daily']), **config.SHARED_ENCODER_CONFIG}
    weekly_config = {'feature_size': len(config.FEATURE_COLUMNS['weekly']), **config.SHARED_ENCODER_CONFIG}
    monthly_config = {'feature_size': len(config.FEATURE_COLUMNS['monthly']), **config.SHARED_ENCODER_CONFIG}
    
    model = MultiEncoderFusionModel(
        daily_config=daily_config,
        weekly_config=weekly_config,
        monthly_config=monthly_config,
        fusion_dim=config.FUSION_DIM,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    # 使用原始的KL散度，但添加数值稳定性处理
    base_criterion = nn.KLDivLoss(reduction='batchmean')
    
    def stable_kl_loss(log_probs, soft_targets):
        """
        数值稳定的KL散度损失函数
        保持概率分布的特性，只是增加稳定性检查
        """
        # 确保软标签是有效的概率分布
        epsilon = 1e-8
        soft_targets = torch.clamp(soft_targets, min=epsilon, max=1.0-epsilon)
        # 重新归一化
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        
        # 确保log_probs是有效的
        log_probs = torch.clamp(log_probs, min=-50, max=0)  # log概率应该 <= 0
        
        # 计算KL散度
        kl_loss = base_criterion(log_probs, soft_targets)
        
        # 最终检查
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print(f"WARNING: KL loss is {kl_loss.item()}, using fallback calculation")
            # 手动计算KL散度作为备选
            probs = torch.exp(log_probs)
            probs = torch.clamp(probs, min=epsilon, max=1.0-epsilon)
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            # KL(P||Q) = sum(P * log(P/Q))
            kl_manual = (soft_targets * (torch.log(soft_targets + epsilon) - torch.log(probs + epsilon))).sum(dim=1).mean()
            return torch.clamp(kl_manual, min=0, max=100)  # 限制损失范围
        
        return kl_loss
    
    criterion = stable_kl_loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    logger.info("模型、损失函数和优化器初始化完成")

    # --- 4. 训练循环 ---
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    best_models = {
        'loss': [(float('inf'), '')] * 3,
        'acc@1': [(0.0, '')] * 1,
        'acc@2': [(0.0, '')] * 1,
        'acc@3': [(0.0, '')] * 1,
    }

    logger.info(f"开始在 {config.DEVICE} 上训练 {config.EPOCHS} 个epoch...")
    
    # 在训练开始前做一次快速的数据检查
    if config.DEBUG_MODE or config.ENABLE_DATA_VALIDATION:
        logger.debug("训练前数据健全检查...")
        sample_batch = next(iter(train_loader))
        logger.debug(f"样本批次形状: 日线={sample_batch['daily'].shape}, 周线={sample_batch['weekly'].shape}, 月线={sample_batch['monthly'].shape}, 标签={sample_batch['label'].shape}")
        
        logger.debug(f"样本批次数据范围:")
        logger.debug(f"  日线: [{sample_batch['daily'].min().item():.6f}, {sample_batch['daily'].max().item():.6f}]")
        logger.debug(f"  周线: [{sample_batch['weekly'].min().item():.6f}, {sample_batch['weekly'].max().item():.6f}]")
        logger.debug(f"  月线: [{sample_batch['monthly'].min().item():.6f}, {sample_batch['monthly'].max().item():.6f}]")
        logger.debug(f"  标签: [{sample_batch['label'].min().item():.6f}, {sample_batch['label'].max().item():.6f}]")
        
        # 检查是否有NaN或Inf
        for name, data in [('日线', sample_batch['daily']), ('周线', sample_batch['weekly']),
                           ('月线', sample_batch['monthly']), ('标签', sample_batch['label'])]:
            nan_count = torch.isnan(data).sum().item()
            inf_count = torch.isinf(data).sum().item()
            if nan_count > 0 or inf_count > 0:
                logger.error(f"{name}包含 {nan_count} 个NaN和 {inf_count} 个Inf值")
    
    for epoch in range(config.EPOCHS):
        if config.DEBUG_MODE:
            logger.debug(f"=== Epoch {epoch+1}/{config.EPOCHS} ===")
        train_loss, train_accs = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_accs = evaluate(model, val_loader, criterion, config.DEVICE)
        
        train_acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_accs.items()])
        val_acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_accs.items()])

        logger.info(f"Epoch {epoch+1:02}/{config.EPOCHS} | "
              f"训练损失: {train_loss:.6f}, 训练准确率: [{train_acc_str}] | "
              f"验证损失: {val_loss:.6f}, 验证准确率: [{val_acc_str}]")
        
        # --- 保存最佳模型逻辑 ---
        def update_best_models(metric_type, value, epoch, is_loss=False):
            key = metric_type
            limit = len(best_models[key])
            
            # For loss, we want the minimum, so we compare negatively
            comparison_val = -value if is_loss else value
            current_best_vals = [(-v[0] if is_loss else v[0]) for v in best_models[key]]

            if comparison_val > min(current_best_vals):
                # Find the one to replace
                idx_to_replace = np.argmin(current_best_vals)
                old_path = best_models[key][idx_to_replace][1]
                if os.path.exists(old_path):
                    os.remove(old_path)
                
                # Add new model
                new_path = os.path.join(config.MODEL_DIR, f"model_best_{key.replace('@','_')}_{idx_to_replace+1}.pth")
                best_models[key][idx_to_replace] = (value, new_path)
                torch.save(model.state_dict(), new_path)
                # 注意：滚动窗口归一化不需要保存scaler，因为预测时会基于最新数据重新计算
                logger.info(f"  -> 保存新的 top-{key} 模型到 {new_path}")

        update_best_models('loss', val_loss, epoch + 1, is_loss=True)
        for acc_key, acc_val in val_accs.items():
            update_best_models(acc_key, acc_val, epoch + 1)

    logger.info("训练完成")

    # --- 5. 在测试集上进行最终评估 ---
    logger.info("开始在测试集上进行最终评估...")
    best_loss_model_path = sorted(best_models['loss'], key=lambda x: x[0])[0][1]
    if not os.path.exists(best_loss_model_path):
        logger.error("未找到最佳模型进行评估，退出")
        return
        
    logger.info(f"加载最佳模型（按损失）: {best_loss_model_path}")
    model.load_state_dict(torch.load(best_loss_model_path))
    
    test_loss, test_accs = evaluate(model, test_loader, criterion, config.DEVICE)
    test_acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_accs.items()])
    
    logger.info(f"最终测试结果 -> 损失: {test_loss:.6f}, 准确率: [{test_acc_str}]")
    logger.info("评估完成")

if __name__ == '__main__':
    main()