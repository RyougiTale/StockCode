#!/usr/bin/env python3
"""
增强版3D训练引擎 - 包含详细的日志记录、相关性测试和性能监控
"""

import torch
import torch.nn.utils as nn_utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import os
from datetime import datetime

try:
    from . import config
    from .logger_config import get_logger
    from .engine_3d import _check_for_anomalies, _check_gradients, _update_topk_accuracy
    from .data_validation import validate_tensor_batch, clean_tensor_batch
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    from logger_config import get_logger
    from engine_3d import _check_for_anomalies, _check_gradients, _update_topk_accuracy
    from data_validation import validate_tensor_batch, clean_tensor_batch

logger = get_logger(__name__)

class TrainingMonitor:
    """训练过程监控器"""
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.metrics_history = defaultdict(list)
        self.correlation_history = defaultdict(list)
        self.prediction_samples = defaultdict(list)
        self.label_samples = defaultdict(list)
        
    def log_epoch_metrics(self, epoch, train_losses, train_accs, val_losses, val_accs):
        """记录每个epoch的指标"""
        # 记录损失
        self.metrics_history['epoch'].append(epoch)
        for key in ['total', 'return', 'sharpe', 'drawdown']:
            self.metrics_history[f'train_loss_{key}'].append(train_losses[key])
            self.metrics_history[f'val_loss_{key}'].append(val_losses[key])
        
        # 记录准确率
        for dim in ['return', 'sharpe', 'drawdown']:
            if dim in train_accs:
                self.metrics_history[f'train_acc_{dim}_top1'].append(train_accs[dim]['acc@1'])
                self.metrics_history[f'val_acc_{dim}_top1'].append(val_accs[dim]['acc@1'])
    
    def save_training_curves(self):
        """保存训练曲线图"""
        if len(self.metrics_history['epoch']) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1 = axes[0, 0]
        epochs = self.metrics_history['epoch']
        for key in ['total', 'return', 'sharpe', 'drawdown']:
            ax1.plot(epochs, self.metrics_history[f'train_loss_{key}'], 
                    label=f'Train {key}', linestyle='-')
            ax1.plot(epochs, self.metrics_history[f'val_loss_{key}'], 
                    label=f'Val {key}', linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2 = axes[0, 1]
        for dim in ['return', 'sharpe', 'drawdown']:
            ax2.plot(epochs, self.metrics_history[f'train_acc_{dim}_top1'], 
                    label=f'Train {dim}', linestyle='-')
            ax2.plot(epochs, self.metrics_history[f'val_acc_{dim}_top1'], 
                    label=f'Val {dim}', linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Top-1 Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 相关性历史（如果有的话）
        ax3 = axes[1, 0]
        if len(self.correlation_history['epoch']) > 0:
            corr_epochs = self.correlation_history['epoch']
            for dim in ['return', 'sharpe', 'drawdown']:
                if f'{dim}_pearson' in self.correlation_history:
                    ax3.plot(corr_epochs, self.correlation_history[f'{dim}_pearson'], 
                            label=f'{dim} Pearson', marker='o')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Pearson Correlation')
            ax3.set_title('Prediction-Label Correlation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No correlation data', 
                    transform=ax3.transAxes, ha='center', va='center')
        
        # 损失比例图
        ax4 = axes[1, 1]
        if len(epochs) > 0:
            return_ratio = np.array(self.metrics_history['val_loss_return']) / np.array(self.metrics_history['val_loss_total'])
            sharpe_ratio = np.array(self.metrics_history['val_loss_sharpe']) / np.array(self.metrics_history['val_loss_total'])
            drawdown_ratio = np.array(self.metrics_history['val_loss_drawdown']) / np.array(self.metrics_history['val_loss_total'])
            
            ax4.plot(epochs, return_ratio, label='Return Loss Ratio', marker='o')
            ax4.plot(epochs, sharpe_ratio, label='Sharpe Loss Ratio', marker='s')
            ax4.plot(epochs, drawdown_ratio, label='Drawdown Loss Ratio', marker='^')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Component Ratio')
            ax4.set_title('Loss Component Analysis')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.model_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {save_path}")

def enhanced_train_one_epoch_3d(model, dataloader, criterion, optimizer, device, 
                               monitor=None, epoch=None, correlation_test_interval=50):
    """
    增强版3D训练函数 - 包含详细的监控和相关性测试
    """
    model.train()
    
    # 损失累计
    total_losses = {'total': 0.0, 'return': 0.0, 'sharpe': 0.0, 'drawdown': 0.0}
    
    # 准确率累计
    correct_k = {
        'return': {1: 0, 2: 0, 3: 0},
        'sharpe': {1: 0, 2: 0, 3: 0},
        'drawdown': {1: 0, 2: 0, 3: 0}
    }
    total_samples = 0
    
    # 用于相关性分析的样本收集
    correlation_samples = {'predictions': defaultdict(list), 'labels': defaultdict(list)}
    
    # 梯度统计
    gradient_norms = []
    batch_losses = []
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch if epoch else '?'}", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        daily_data = batch['daily'].to(device)
        weekly_data = batch['weekly'].to(device)
        monthly_data = batch['monthly'].to(device)
        
        # 3D标签
        labels_3d = {}
        for dim in ['return', 'sharpe', 'drawdown']:
            labels_3d[dim] = batch['labels_3d'][dim].to(device)
        
        # 数据验证 - 检测NaN/Inf
        input_tensors = {
            'daily': daily_data,
            'weekly': weekly_data,
            'monthly': monthly_data
        }
        input_tensors.update(labels_3d)
        
        if not validate_tensor_batch(input_tensors, batch_idx):
            logger.warning(f"批次 {batch_idx} 包含无效数据，跳过")
            continue
        
        # 前向传播
        outputs = model(daily_data, weekly_data, monthly_data)
        
        # 数值检查
        if _check_for_anomalies(outputs, labels_3d, batch_idx):
            continue
        
        # 计算损失
        losses = criterion(outputs, labels_3d)
        
        # 检查损失有效性
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            logger.warning(f"Epoch {epoch}, Batch {batch_idx}: 跳过NaN/Inf损失")
            continue
        
        # 收集样本用于相关性分析（每隔一定batch采样）
        if batch_idx % correlation_test_interval == 0:
            with torch.no_grad():
                for dim in ['return', 'sharpe', 'drawdown']:
                    # 预测概率 (使用exp因为模型输出log_softmax)
                    pred_probs = torch.exp(outputs[dim]).cpu().numpy()
                    # 真实标签概率
                    true_probs = labels_3d[dim].cpu().numpy()
                    
                    correlation_samples['predictions'][dim].extend(pred_probs)
                    correlation_samples['labels'][dim].extend(true_probs)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total'].backward()
        
        # 梯度裁剪和统计
        if hasattr(config, 'GRAD_CLIP_NORM') and config.GRAD_CLIP_NORM > 0:
            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            gradient_norms.append(grad_norm.item())
        
        # 检查梯度
        if _check_gradients(model):
            continue
        
        # 更新权重
        optimizer.step()
        
        # 统计
        batch_size = daily_data.size(0)
        total_samples += batch_size
        
        for key, loss_val in losses.items():
            total_losses[key] += loss_val.item()
        
        batch_losses.append(losses['total'].item())
        
        # 计算准确率
        for dim in ['return', 'sharpe', 'drawdown']:
            _update_topk_accuracy(outputs[dim], labels_3d[dim], correct_k[dim])
        
        # 更新进度条
        current_loss = losses['total'].item()
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Grad': f'{gradient_norms[-1]:.3f}' if gradient_norms else 'N/A'
        })
    
    # 计算平均指标
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    avg_accs = {}
    for dim in ['return', 'sharpe', 'drawdown']:
        avg_accs[dim] = {}
        for k in [1, 2, 3]:
            avg_accs[dim][f'acc@{k}'] = correct_k[dim][k] / total_samples if total_samples > 0 else 0.0
    
    # 相关性分析
    correlations = {}
    if epoch and epoch % 10 == 0:  # 每10个epoch计算一次相关性
        logger.info(f"Epoch {epoch}: 计算预测-标签相关性...")
        correlations = calculate_prediction_correlations(correlation_samples)
        
        # 记录相关性到监控器
        if monitor:
            monitor.correlation_history['epoch'].append(epoch)
            for dim in ['return', 'sharpe', 'drawdown']:
                if dim in correlations:
                    monitor.correlation_history[f'{dim}_pearson'].append(correlations[dim]['pearson'])
                    monitor.correlation_history[f'{dim}_spearman'].append(correlations[dim]['spearman'])
    
    # 训练统计日志
    if epoch and epoch % 10 == 0:
        log_detailed_training_stats(epoch, avg_losses, avg_accs, correlations, 
                                  gradient_norms, batch_losses)
    
    return avg_losses, avg_accs

def calculate_prediction_correlations(correlation_samples):
    """计算预测与标签之间的相关性"""
    correlations = {}
    
    for dim in ['return', 'sharpe', 'drawdown']:
        if dim not in correlation_samples['predictions'] or len(correlation_samples['predictions'][dim]) == 0:
            continue
            
        # 转换为numpy数组
        pred_probs = np.array(correlation_samples['predictions'][dim])
        true_probs = np.array(correlation_samples['labels'][dim])
        
        if len(pred_probs) == 0 or len(true_probs) == 0:
            continue
        
        # 计算期望值（概率分布的期望）
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # 相对中心点
        
        pred_expectations = np.sum(pred_probs * centers, axis=1)
        true_expectations = np.sum(true_probs * centers, axis=1)
        
        # 计算相关系数
        try:
            pearson_corr, pearson_p = pearsonr(pred_expectations, true_expectations)
            spearman_corr, spearman_p = spearmanr(pred_expectations, true_expectations)
            
            correlations[dim] = {
                'pearson': pearson_corr,
                'pearson_p': pearson_p,
                'spearman': spearman_corr,
                'spearman_p': spearman_p,
                'sample_size': len(pred_expectations)
            }
        except Exception as e:
            logger.warning(f"计算{dim}相关性时出错: {e}")
            correlations[dim] = {'pearson': 0.0, 'spearman': 0.0, 'sample_size': 0}
    
    return correlations

def log_detailed_training_stats(epoch, losses, accs, correlations, gradient_norms, batch_losses):
    """记录详细的训练统计信息"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Epoch {epoch} 详细统计")
    logger.info(f"{'='*60}")
    
    # 损失统计
    logger.info("损失分析:")
    for key, value in losses.items():
        logger.info(f"  {key.upper():>10}: {value:.6f}")
    
    # 准确率统计
    logger.info("\n准确率分析:")
    for dim in ['return', 'sharpe', 'drawdown']:
        if dim in accs:
            logger.info(f"  {dim.upper():>10}: Top1={accs[dim]['acc@1']:.4f}, "
                       f"Top2={accs[dim]['acc@2']:.4f}, Top3={accs[dim]['acc@3']:.4f}")
    
    # 相关性统计
    if correlations:
        logger.info("\n预测-标签相关性:")
        for dim in ['return', 'sharpe', 'drawdown']:
            if dim in correlations:
                corr_info = correlations[dim]
                logger.info(f"  {dim.upper():>10}: Pearson={corr_info['pearson']:.4f} "
                           f"(p={corr_info['pearson_p']:.4f}), "
                           f"Spearman={corr_info['spearman']:.4f} "
                           f"(n={corr_info['sample_size']})")
    
    # 梯度统计
    if gradient_norms:
        grad_mean = np.mean(gradient_norms)
        grad_std = np.std(gradient_norms)
        grad_max = np.max(gradient_norms)
        logger.info(f"\n梯度统计: 均值={grad_mean:.4f}, 标准差={grad_std:.4f}, 最大值={grad_max:.4f}")
    
    # 批次损失统计
    if batch_losses:
        loss_mean = np.mean(batch_losses)
        loss_std = np.std(batch_losses)
        loss_min = np.min(batch_losses)
        loss_max = np.max(batch_losses)
        logger.info(f"批次损失: 均值={loss_mean:.4f}, 标准差={loss_std:.4f}, "
                   f"范围=[{loss_min:.4f}, {loss_max:.4f}]")
    
    logger.info(f"{'='*60}\n")

def enhanced_evaluate_3d(model, dataloader, criterion, device, epoch=None, monitor=None):
    """增强版3D验证函数"""
    model.eval()
    
    total_losses = {'total': 0.0, 'return': 0.0, 'sharpe': 0.0, 'drawdown': 0.0}
    correct_k = {
        'return': {1: 0, 2: 0, 3: 0},
        'sharpe': {1: 0, 2: 0, 3: 0},
        'drawdown': {1: 0, 2: 0, 3: 0}
    }
    total_samples = 0
    
    # 收集样本用于分析
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch if epoch else '?'}", leave=False)
        
        for batch in progress_bar:
            daily_data = batch['daily'].to(device)
            weekly_data = batch['weekly'].to(device)
            monthly_data = batch['monthly'].to(device)
            
            labels_3d = {}
            for dim in ['return', 'sharpe', 'drawdown']:
                labels_3d[dim] = batch['labels_3d'][dim].to(device)
            
            outputs = model(daily_data, weekly_data, monthly_data)
            losses = criterion(outputs, labels_3d)
            
            # 统计
            batch_size = daily_data.size(0)
            total_samples += batch_size
            
            for key, loss_val in losses.items():
                total_losses[key] += loss_val.item()
            
            for dim in ['return', 'sharpe', 'drawdown']:
                _update_topk_accuracy(outputs[dim], labels_3d[dim], correct_k[dim])
                
                # 收集预测和标签
                pred_probs = torch.exp(outputs[dim]).cpu().numpy()
                true_probs = labels_3d[dim].cpu().numpy()
                all_predictions[dim].extend(pred_probs)
                all_labels[dim].extend(true_probs)
    
    # 计算平均指标
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    avg_accs = {}
    for dim in ['return', 'sharpe', 'drawdown']:
        avg_accs[dim] = {}
        for k in [1, 2, 3]:
            avg_accs[dim][f'acc@{k}'] = correct_k[dim][k] / total_samples if total_samples > 0 else 0.0
    
    # 计算验证集上的相关性
    val_correlations = {}
    if epoch and epoch % 10 == 0:
        correlation_samples = {'predictions': all_predictions, 'labels': all_labels}
        val_correlations = calculate_prediction_correlations(correlation_samples)
        
        logger.info(f"Epoch {epoch} 验证集相关性:")
        for dim in ['return', 'sharpe', 'drawdown']:
            if dim in val_correlations:
                corr_info = val_correlations[dim]
                logger.info(f"  {dim}: Pearson={corr_info['pearson']:.4f}, "
                           f"Spearman={corr_info['spearman']:.4f}")
    
    return avg_losses, avg_accs