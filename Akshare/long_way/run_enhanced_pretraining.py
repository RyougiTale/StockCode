#!/usr/bin/env python3
"""
增强版预训练脚本 - 包含详细的日志记录、相关性测试和性能监控
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

# 添加父目录到路径以便导入
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# 导入我们的模块
try:
    from . import config
    from .model_3d import create_3d_model, Multi3DLoss
    from .dataset_3d import create_3d_datasets_with_distribution
    from .data_utils import get_all_samples
    from .enhanced_engine_3d import (
        enhanced_train_one_epoch_3d, enhanced_evaluate_3d, 
        TrainingMonitor, log_detailed_training_stats
    )
    from .logger_config import get_logger, setup_logging
    from .data_validation import validate_sample_data, check_data_distribution
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    from model_3d import create_3d_model, Multi3DLoss
    from dataset_3d import create_3d_datasets_with_distribution
    from data_utils import get_all_samples
    from enhanced_engine_3d import (
        enhanced_train_one_epoch_3d, enhanced_evaluate_3d, 
        TrainingMonitor, log_detailed_training_stats
    )
    from logger_config import get_logger, setup_logging
    from data_validation import validate_sample_data, check_data_distribution

# 确保我们在预训练模式
assert config.TRAINING_PHASE == "pretraining", "请在config.py中设置TRAINING_PHASE为'pretraining'"

# 初始化日志系统
setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

def analyze_dataset_distribution(train_dataset, val_dataset, test_dataset, save_dir):
    """分析数据集中标签的分布"""
    logger.info("分析数据集标签分布...")
    
    def collect_labels(dataset, name):
        labels_data = {'return': [], 'sharpe': [], 'drawdown': []}
        
        # 采样一部分数据进行分析（避免内存过大）
        sample_size = min(1000, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        for idx in tqdm(indices, desc=f"分析{name}标签"):
            try:
                sample = dataset[idx]
                for dim in ['return', 'sharpe', 'drawdown']:
                    labels_data[dim].append(sample['labels_3d'][dim].numpy())
            except Exception as e:
                logger.warning(f"分析样本{idx}时出错: {e}")
                continue
        
        return labels_data
    
    # 收集各数据集的标签
    train_labels = collect_labels(train_dataset, "训练集")
    val_labels = collect_labels(val_dataset, "验证集")
    test_labels = collect_labels(test_dataset, "测试集")
    
    # 可视化标签分布
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    for i, dim in enumerate(['return', 'sharpe', 'drawdown']):
        for j, (data, name) in enumerate([(train_labels, '训练集'), 
                                        (val_labels, '验证集'), 
                                        (test_labels, '测试集')]):
            ax = axes[i, j]
            
            if len(data[dim]) > 0:
                # 计算每个类别的平均概率
                probs = np.array(data[dim])
                mean_probs = probs.mean(axis=0)
                std_probs = probs.std(axis=0)
                
                # 绘制条形图
                categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                bars = ax.bar(categories, mean_probs, yerr=std_probs, 
                            capsize=5, alpha=0.7)
                
                ax.set_title(f'{dim.title()} - {name}')
                ax.set_ylabel('平均概率')
                ax.set_ylim(0, 1)
                
                # 添加数值标签
                for bar, prob in zip(bars, mean_probs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
                
                # 检查是否过于均匀
                entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
                max_entropy = np.log(5)  # 5个类别的最大熵
                ax.text(0.02, 0.98, f'熵: {entropy:.3f}/{max_entropy:.3f}', 
                       transform=ax.transAxes, va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'无{name}数据', 
                       transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'dataset_label_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"数据集标签分布图已保存: {save_path}")
    
    # 记录分布统计
    logger.info("\n数据集标签分布统计:")
    for dim in ['return', 'sharpe', 'drawdown']:
        logger.info(f"\n{dim.upper()} 维度:")
        for name, data in [('训练集', train_labels), ('验证集', val_labels), ('测试集', test_labels)]:
            if len(data[dim]) > 0:
                probs = np.array(data[dim])
                mean_probs = probs.mean(axis=0)
                entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
                logger.info(f"  {name}: 平均概率={mean_probs}, 熵={entropy:.3f}")

def run_initial_model_test(model, train_loader, criterion, device):
    """在训练开始前测试模型的初始状态"""
    logger.info("测试模型初始状态...")
    
    model.eval()
    test_samples = 5  # 测试几个batch
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= test_samples:
                break
                
            daily_data = batch['daily'].to(device)
            weekly_data = batch['weekly'].to(device)
            monthly_data = batch['monthly'].to(device)
            
            labels_3d = {}
            for dim in ['return', 'sharpe', 'drawdown']:
                labels_3d[dim] = batch['labels_3d'][dim].to(device)
            
            # 前向传播
            outputs = model(daily_data, weekly_data, monthly_data)
            losses = criterion(outputs, labels_3d)
            
            logger.info(f"初始测试 Batch {batch_idx+1}:")
            logger.info(f"  总损失: {losses['total'].item():.4f}")
            for dim in ['return', 'sharpe', 'drawdown']:
                logger.info(f"  {dim} 损失: {losses[dim].item():.4f}")
                
                # 分析预测分布
                pred_probs = torch.exp(outputs[dim]).cpu().numpy()
                mean_pred = pred_probs.mean(axis=0)
                logger.info(f"  {dim} 平均预测: {mean_pred}")

def enhanced_save_best_models(model, val_losses, val_accs, best_models, save_dir, epoch, 
                            correlations=None):
    """增强版模型保存，包含相关性信息"""
    min_epoch_to_save = 10  # 前10个epoch不保存
    
    if epoch < min_epoch_to_save:
        return
    
    # 保存最佳损失模型
    current_loss = val_losses['total']
    current_best_losses = [v[0] for v in best_models['total_loss']]
    
    if current_loss < max(current_best_losses):
        idx_to_replace = np.argmax(current_best_losses)
        old_path = best_models['total_loss'][idx_to_replace][1]
        
        new_path = os.path.join(save_dir, f"best_loss_top_{idx_to_replace+1}.pth")
        
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
        torch.save(model.state_dict(), new_path)
        
        best_models['total_loss'][idx_to_replace] = (current_loss, new_path)
        
        # 记录详细信息
        correlation_info = ""
        if correlations:
            corr_values = [correlations.get(dim, {}).get('pearson', 0.0) 
                          for dim in ['return', 'sharpe', 'drawdown']]
            correlation_info = f", 相关性=[{corr_values[0]:.3f}, {corr_values[1]:.3f}, {corr_values[2]:.3f}]"
        
        logger.info(f"  🎯 保存Top-{idx_to_replace+1}损失模型 (Epoch {epoch}): "
                   f"损失 {max(current_best_losses):.6f} -> {current_loss:.6f}{correlation_info}")
    
    # 保存最佳准确率模型
    for dim in ['return', 'sharpe', 'drawdown']:
        key = f'{dim}_acc@1'
        if key in best_models and dim in val_accs:
            current_acc = val_accs[dim]['acc@1']
            best_acc = best_models[key][0][0]
            
            if current_acc > best_acc and current_acc > 0.21:  # 略高于随机(0.2)
                old_path = best_models[key][0][1]
                new_path = os.path.join(save_dir, f"best_{dim}_acc.pth")
                
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                torch.save(model.state_dict(), new_path)
                
                best_models[key][0] = (current_acc, new_path)
                
                correlation_info = ""
                if correlations and dim in correlations:
                    correlation_info = f", 相关性={correlations[dim]['pearson']:.3f}"
                
                logger.info(f"  🏆 保存最佳{dim}准确率模型 (Epoch {epoch}): "
                           f"准确率 {best_acc:.4f} -> {current_acc:.4f}{correlation_info}")

def main():
    """增强版预训练主函数"""
    logger.info("🚀 开始增强版3D模型预训练...")
    logger.info(f"训练配置: 设备={config.DEVICE}, 批次大小={config.BATCH_SIZE}, "
               f"学习率={config.LEARNING_RATE}, 轮数={config.EPOCHS}")
    
    # 1. 准备数据
    logger.info("📊 数据准备阶段...")
    all_samples_raw, scalers = get_all_samples(config.STOCK_CODES)
    if not all_samples_raw:
        logger.error("❌ 无法获取样本数据")
        return
    
    # 数据验证和清理
    logger.info("🔍 验证和清理数据...")
    all_samples, removed_count = validate_sample_data(all_samples_raw)
    logger.info(f"数据验证完成: 移除 {removed_count} 个无效样本，保留 {len(all_samples)} 个有效样本")
    
    if not all_samples:
        logger.error("❌ 验证后无有效数据")
        return
    
    # 检查数据分布
    check_data_distribution(all_samples, sample_size=min(1000, len(all_samples)))
    
    logger.info(f"总有效样本数: {len(all_samples):,}")
    
    # 时间筛选
    if config.TRAINING_YEARS is not None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * config.TRAINING_YEARS)
        
        original_count = len(all_samples)
        all_samples = [s for s in all_samples if s['date'].to_pydatetime() >= start_date]
        logger.info(f"时间筛选: {len(all_samples):,}/{original_count:,} 样本 "
                   f"(最近{config.TRAINING_YEARS}年)")
    
    # 2. 创建数据集
    logger.info("🏗️ 创建3D数据集...")
    train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
        all_samples,
        train_ratio=0.75,
        val_ratio=0.15,
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    logger.info(f"数据集划分: 训练={len(train_dataset):,}, 验证={len(val_dataset):,}, "
               f"测试={len(test_dataset):,}")
    logger.info(f"批次数量: 训练={len(train_loader)}, 验证={len(val_loader)}, "
               f"测试={len(test_loader)}")
    
    # 分析数据集分布
    model_dir = os.path.join(config.MODEL_DIR, "enhanced_pretraining")
    os.makedirs(model_dir, exist_ok=True)
    analyze_dataset_distribution(train_dataset, val_dataset, test_dataset, model_dir)
    
    # 3. 创建模型
    logger.info("🧠 创建3D模型...")
    model = create_3d_model(config).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 4. 损失函数和优化器
    criterion = Multi3DLoss(weights={
        'return': 1.0,
        'sharpe': 0.8, 
        'drawdown': 0.6
    })
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
    )
    
    # 5. 训练监控器
    monitor = TrainingMonitor(model_dir)
    
    # 6. 初始模型测试
    run_initial_model_test(model, train_loader, criterion, config.DEVICE)
    
    # 7. 最佳模型跟踪
    best_models = {
        'total_loss': [(float('inf'), '')] * 3,
        'return_acc@1': [(0.0, '')],
        'sharpe_acc@1': [(0.0, '')], 
        'drawdown_acc@1': [(0.0, '')]
    }
    
    # 8. 训练循环
    logger.info("🎯 开始训练...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 100
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = datetime.now()
        
        # 训练
        train_losses, train_accs = enhanced_train_one_epoch_3d(
            model, train_loader, criterion, optimizer, config.DEVICE,
            monitor=monitor, epoch=epoch, correlation_test_interval=50
        )
        
        # 验证
        val_losses, val_accs = enhanced_evaluate_3d(
            model, val_loader, criterion, config.DEVICE, 
            epoch=epoch, monitor=monitor
        )
        
        # 学习率调度
        scheduler.step(val_losses['total'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到监控器
        monitor.log_epoch_metrics(epoch, train_losses, train_accs, val_losses, val_accs)
        
        # 打印训练结果
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        logger.info(f"\n📈 Epoch {epoch:3d}/{config.EPOCHS} (耗时 {epoch_time:.1f}s, LR={current_lr:.2e})")
        logger.info(f"   训练损失: 总={train_losses['total']:.4f}, "
                   f"回报={train_losses['return']:.4f}, "
                   f"夏普={train_losses['sharpe']:.4f}, "
                   f"回撤={train_losses['drawdown']:.4f}")
        logger.info(f"   验证损失: 总={val_losses['total']:.4f}, "
                   f"回报={val_losses['return']:.4f}, "
                   f"夏普={val_losses['sharpe']:.4f}, "
                   f"回撤={val_losses['drawdown']:.4f}")
        logger.info(f"   验证准确率: "
                   f"回报={val_accs['return']['acc@1']:.3f}, "
                   f"夏普={val_accs['sharpe']['acc@1']:.3f}, "
                   f"回撤={val_accs['drawdown']['acc@1']:.3f}")
        
        # 保存最佳模型
        enhanced_save_best_models(model, val_losses, val_accs, best_models, 
                                model_dir, epoch)
        
        # 早停检查
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            logger.info(f"💤 早停触发 (patience={max_patience})")
            break
        
        # 定期保存训练曲线
        if epoch % 50 == 0:
            monitor.save_training_curves()
    
    # 9. 保存最终训练曲线
    monitor.save_training_curves()
    
    # 10. 最终测试
    logger.info("🏁 最终测试评估...")
    best_loss_models = sorted(best_models['total_loss'], key=lambda x: x[0])
    best_model_path = best_loss_models[0][1]
    
    if os.path.exists(best_model_path):
        logger.info(f"加载最佳模型: {os.path.basename(best_model_path)}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        
        test_losses, test_accs = enhanced_evaluate_3d(model, test_loader, criterion, config.DEVICE)
        
        logger.info("🎊 最终测试结果:")
        logger.info(f"   测试损失: 总={test_losses['total']:.4f}, "
                   f"回报={test_losses['return']:.4f}, "
                   f"夏普={test_losses['sharpe']:.4f}, "
                   f"回撤={test_losses['drawdown']:.4f}")
        logger.info(f"   测试准确率: "
                   f"回报={test_accs['return']['acc@1']:.3f}, "
                   f"夏普={test_accs['sharpe']['acc@1']:.3f}, "
                   f"回撤={test_accs['drawdown']['acc@1']:.3f}")
    
    logger.info("✅ 增强版预训练完成!")
    logger.info(f"📁 所有文件保存在: {model_dir}")

if __name__ == '__main__':
    if config.TRAINING_PHASE != "pretraining":
        logger.error(f"❌ 当前训练阶段为 '{config.TRAINING_PHASE}'，请设置为'pretraining'")
        exit(1)
    
    main()