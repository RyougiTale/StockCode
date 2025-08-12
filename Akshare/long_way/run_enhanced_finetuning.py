#!/usr/bin/env python3
"""
增强版微调脚本 - 包含详细的相关性监控和预测准确性分析
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

# 添加父目录到路径以便导入
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

try:
    from . import config
    from .model_3d import create_3d_model, Multi3DLoss
    from .dataset_3d import create_3d_datasets_with_distribution
    from .data_utils import get_all_samples
    from .enhanced_engine_3d import (
        enhanced_train_one_epoch_3d, enhanced_evaluate_3d, 
        TrainingMonitor, calculate_prediction_correlations
    )
    from .draw_3d_long_term import predict_3d_long_term
    from .logger_config import get_logger, setup_logging
except ImportError:
    import config
    from model_3d import create_3d_model, Multi3DLoss
    from dataset_3d import create_3d_datasets_with_distribution
    from data_utils import get_all_samples
    from enhanced_engine_3d import (
        enhanced_train_one_epoch_3d, enhanced_evaluate_3d, 
        TrainingMonitor, calculate_prediction_correlations
    )
    from draw_3d_long_term import predict_3d_long_term
    from logger_config import get_logger, setup_logging

# 确保我们在微调模式
assert config.TRAINING_PHASE == "finetuning", "请在config.py中设置TRAINING_PHASE为'finetuning'"

# 初始化日志系统
setup_logging(log_level=config.LOGGING_LEVEL)
logger = get_logger(__name__)

def test_prediction_accuracy_during_training(model, stock_code, model_dir, epoch):
    """在训练过程中测试预测准确性"""
    logger.info(f"Epoch {epoch}: 测试 {stock_code} 的预测准确性...")
    
    # 保存当前模型
    temp_model_path = os.path.join(model_dir, f"temp_epoch_{epoch}.pth")
    torch.save(model.state_dict(), temp_model_path)
    
    try:
        # 运行预测测试（使用1年数据快速测试）
        df = predict_3d_long_term(stock_code, temp_model_path, years=1)
        
        if df.empty:
            logger.warning(f"Epoch {epoch}: 无法获取 {stock_code} 的预测数据")
            return None
        
        # 只分析有实际数据的部分
        df_analysis = df[df['actual_return'].notna()].copy()
        
        if df_analysis.empty:
            logger.warning(f"Epoch {epoch}: 无有效的对比数据")
            return None
        
        # 计算相关性
        correlations = {}
        for metric, name in [('return', '收益率'), ('sharpe', '夏普比率'), ('drawdown', '最大回撤')]:
            actual_col = f'actual_{metric}'
            pred_col = f'pred_{metric}_full'
            
            if actual_col in df_analysis.columns and pred_col in df_analysis.columns:
                corr = df_analysis[actual_col].corr(df_analysis[pred_col])
                correlations[metric] = corr
                
                # 方向准确率（仅对收益率）
                if metric == 'return':
                    actual_direction = np.sign(df_analysis[actual_col])
                    pred_direction = np.sign(df_analysis[pred_col])
                    direction_accuracy = np.mean(actual_direction == pred_direction)
                    correlations[f'{metric}_direction_acc'] = direction_accuracy
        
        logger.info(f"Epoch {epoch} 预测准确性:")
        for key, value in correlations.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return correlations
        
    except Exception as e:
        logger.warning(f"Epoch {epoch}: 测试预测准确性时出错: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

def load_pretrained_model(model, pretrained_path):
    """加载预训练模型权重"""
    if not os.path.exists(pretrained_path):
        logger.warning(f"预训练模型文件不存在: {pretrained_path}")
        return False
    
    try:
        pretrained_state = torch.load(pretrained_path, map_location=config.DEVICE)
        model.load_state_dict(pretrained_state)
        logger.info(f"✅ 成功加载预训练模型: {pretrained_path}")
        return True
    except Exception as e:
        logger.error(f"❌ 加载预训练模型失败: {e}")
        return False

def analyze_target_stock_data(stock_code, all_samples):
    """分析目标股票的数据特点"""
    logger.info(f"📊 分析目标股票 {stock_code} 的数据特点...")
    
    # 筛选目标股票的样本
    stock_samples = [s for s in all_samples if s.get('stock_code', '') == stock_code]
    
    if not stock_samples:
        logger.warning(f"未找到股票 {stock_code} 的样本数据")
        return
    
    logger.info(f"股票 {stock_code} 样本数: {len(stock_samples)}")
    
    # 分析时间分布
    dates = [s['date'] for s in stock_samples]
    if dates:
        earliest = min(dates)
        latest = max(dates)
        logger.info(f"时间范围: {earliest.date()} 到 {latest.date()}")
        
        # 按年份统计
        year_counts = {}
        for date in dates:
            year = date.year
            year_counts[year] = year_counts.get(year, 0) + 1
        
        logger.info("按年份分布:")
        for year in sorted(year_counts.keys()):
            logger.info(f"  {year}: {year_counts[year]} 样本")
    
    # 分析价格特征
    prices = []
    returns = []
    for sample in stock_samples[:100]:  # 取前100个样本分析
        if 'future_prices' in sample and len(sample['future_prices']) > 1:
            price_series = sample['future_prices']
            prices.extend(price_series)
            
            # 计算收益率
            for i in range(1, len(price_series)):
                ret = (price_series[i] - price_series[i-1]) / price_series[i-1]
                returns.append(ret)
    
    if prices and returns:
        logger.info(f"价格统计: 均值={np.mean(prices):.2f}, "
                   f"标准差={np.std(prices):.2f}, "
                   f"范围=[{np.min(prices):.2f}, {np.max(prices):.2f}]")
        logger.info(f"收益率统计: 均值={np.mean(returns):.4f}, "
                   f"标准差={np.std(returns):.4f}, "
                   f"范围=[{np.min(returns):.4f}, {np.max(returns):.4f}]")

def main():
    """增强版微调主函数"""
    target_stock = config.FINETUNING_CONFIG["target_stock"]
    
    logger.info(f"🎯 开始增强版微调 - 目标股票: {target_stock}")
    logger.info(f"配置: 设备={config.DEVICE}, 批次大小={config.BATCH_SIZE}, "
               f"学习率={config.LEARNING_RATE}, 轮数={config.EPOCHS}")
    
    # 1. 数据准备
    logger.info("📊 数据准备...")
    all_samples, scalers = get_all_samples([target_stock])  # 只使用目标股票
    if not all_samples:
        logger.error(f"❌ 无法获取股票 {target_stock} 的数据")
        return
    
    logger.info(f"目标股票样本数: {len(all_samples):,}")
    
    # 分析目标股票数据
    analyze_target_stock_data(target_stock, all_samples)
    
    # 时间筛选
    if config.TRAINING_YEARS is not None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * config.TRAINING_YEARS)
        
        original_count = len(all_samples)
        all_samples = [s for s in all_samples if s['date'].to_pydatetime() >= start_date]
        logger.info(f"时间筛选: {len(all_samples):,}/{original_count:,} 样本")
    
    # 2. 创建数据集
    logger.info("🏗️ 创建微调数据集...")
    train_dataset, val_dataset, test_dataset, stock_distributions = create_3d_datasets_with_distribution(
        all_samples,
        train_ratio=0.8,  # 微调时可以用更多训练数据
        val_ratio=0.1,
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    logger.info(f"数据集: 训练={len(train_dataset):,}, 验证={len(val_dataset):,}, "
               f"测试={len(test_dataset):,}")
    
    # 3. 创建模型
    logger.info("🧠 创建模型...")
    model = create_3d_model(config).to(config.DEVICE)
    
    # 4. 加载预训练模型
    if config.FINETUNING_CONFIG["use_pretrained_model"]:
        pretrained_paths = [
            os.path.join(config.MODEL_DIR, "enhanced_pretraining", "best_loss_top_1.pth"),
            os.path.join(config.MODEL_DIR, "best_loss_top_1.pth"),
            os.path.join(config.MODEL_DIR, "3d_models", "best_loss_top_1.pth")
        ]
        
        loaded = False
        for path in pretrained_paths:
            if load_pretrained_model(model, path):
                loaded = True
                break
        
        if not loaded:
            logger.warning("⚠️ 未找到预训练模型，将从随机初始化开始")
    
    # 5. 损失函数和优化器
    criterion = Multi3DLoss(weights={
        'return': 1.2,  # 微调时稍微强调收益率预测
        'sharpe': 0.9,
        'drawdown': 0.7
    })
    
    # 微调用更小的学习率
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 6. 训练监控
    model_dir = os.path.join(config.MODEL_DIR, f"finetuning_{target_stock}")
    os.makedirs(model_dir, exist_ok=True)
    monitor = TrainingMonitor(model_dir)
    
    # 7. 最佳模型跟踪
    best_models = {
        'total_loss': [(float('inf'), '')],
        'prediction_corr': [(0.0, '')],  # 基于预测相关性的模型保存
    }
    
    # 8. 训练循环
    logger.info("🎯 开始微调训练...")
    
    prediction_accuracies = []  # 记录预测准确性变化
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = datetime.now()
        
        # 训练
        train_losses, train_accs = enhanced_train_one_epoch_3d(
            model, train_loader, criterion, optimizer, config.DEVICE,
            monitor=monitor, epoch=epoch, correlation_test_interval=20  # 更频繁的相关性测试
        )
        
        # 验证
        val_losses, val_accs = enhanced_evaluate_3d(
            model, val_loader, criterion, config.DEVICE, 
            epoch=epoch, monitor=monitor
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到监控器
        monitor.log_epoch_metrics(epoch, train_losses, train_accs, val_losses, val_accs)
        
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # 每10个epoch测试一次预测准确性
        prediction_corr = None
        if epoch % 10 == 0:
            pred_acc = test_prediction_accuracy_during_training(model, target_stock, model_dir, epoch)
            if pred_acc:
                prediction_accuracies.append((epoch, pred_acc))
                prediction_corr = pred_acc.get('return', 0.0)  # 使用收益率相关性作为主要指标
        
        # 打印结果
        logger.info(f"\n📈 Epoch {epoch:3d}/{config.EPOCHS} (耗时 {epoch_time:.1f}s, LR={current_lr:.2e})")
        logger.info(f"   训练: Loss={train_losses['total']:.4f}, "
                   f"回报Acc={train_accs['return']['acc@1']:.3f}")
        logger.info(f"   验证: Loss={val_losses['total']:.4f}, "
                   f"回报Acc={val_accs['return']['acc@1']:.3f}")
        
        if prediction_corr is not None:
            logger.info(f"   预测相关性: {prediction_corr:.4f}")
        
        # 保存最佳模型
        # 基于验证损失
        if val_losses['total'] < best_models['total_loss'][0][0]:
            old_path = best_models['total_loss'][0][1]
            new_path = os.path.join(model_dir, f"best_loss_finetuned_{target_stock}.pth")
            
            if old_path and os.path.exists(old_path):
                os.remove(old_path)
            torch.save(model.state_dict(), new_path)
            best_models['total_loss'][0] = (val_losses['total'], new_path)
            
            logger.info(f"  💾 保存最佳损失模型: {os.path.basename(new_path)}")
        
        # 基于预测相关性
        if prediction_corr and prediction_corr > best_models['prediction_corr'][0][0]:
            old_path = best_models['prediction_corr'][0][1]
            new_path = os.path.join(model_dir, f"best_corr_finetuned_{target_stock}.pth")
            
            if old_path and os.path.exists(old_path):
                os.remove(old_path)
            torch.save(model.state_dict(), new_path)
            best_models['prediction_corr'][0] = (prediction_corr, new_path)
            
            logger.info(f"  🎯 保存最佳相关性模型: {os.path.basename(new_path)} (相关性={prediction_corr:.4f})")
    
    # 9. 保存训练曲线和预测准确性曲线
    monitor.save_training_curves()
    
    # 绘制预测准确性变化
    if prediction_accuracies:
        epochs = [x[0] for x in prediction_accuracies]
        return_corrs = [x[1].get('return', 0) for x in prediction_accuracies]
        direction_accs = [x[1].get('return_direction_acc', 0) for x in prediction_accuracies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(epochs, return_corrs, 'o-', label='收益率相关性')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Pearson相关系数')
        ax1.set_title(f'{target_stock} - 预测相关性变化')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax2.plot(epochs, direction_accs, 'o-', label='方向准确率', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('方向准确率')
        ax2.set_title(f'{target_stock} - 方向预测准确率')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='随机水平')
        ax2.legend()
        
        plt.tight_layout()
        pred_acc_path = os.path.join(model_dir, 'prediction_accuracy_evolution.png')
        plt.savefig(pred_acc_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"预测准确性变化图已保存: {pred_acc_path}")
    
    # 10. 最终测试
    logger.info("🏁 最终测试...")
    
    # 使用最佳相关性模型进行测试（如果有的话）
    best_model_path = best_models['prediction_corr'][0][1] if best_models['prediction_corr'][0][1] else best_models['total_loss'][0][1]
    
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        logger.info(f"使用最佳模型: {os.path.basename(best_model_path)}")
        
        # 测试集评估
        test_losses, test_accs = enhanced_evaluate_3d(model, test_loader, criterion, config.DEVICE)
        logger.info("测试集结果:")
        logger.info(f"  损失: {test_losses['total']:.4f}")
        logger.info(f"  准确率: 回报={test_accs['return']['acc@1']:.3f}, "
                   f"夏普={test_accs['sharpe']['acc@1']:.3f}, "
                   f"回撤={test_accs['drawdown']['acc@1']:.3f}")
        
        # 最终预测准确性测试
        final_pred_acc = test_prediction_accuracy_during_training(model, target_stock, model_dir, "final")
        if final_pred_acc:
            logger.info("最终预测准确性:")
            for key, value in final_pred_acc.items():
                logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"✅ 微调完成! 所有文件保存在: {model_dir}")

if __name__ == '__main__':
    if config.TRAINING_PHASE != "finetuning":
        logger.error(f"❌ 当前训练阶段为 '{config.TRAINING_PHASE}'，请设置为'finetuning'")
        exit(1)
    
    main()