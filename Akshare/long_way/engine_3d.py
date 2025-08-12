import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm
try:
    from . import config
    from .logger_config import get_logger
except ImportError:
    import config
    from logger_config import get_logger

# 获取日志记录器
logger = get_logger(__name__)

def train_one_epoch_3d(model, dataloader, criterion, optimizer, device, grad_clip_norm=1.0):
    """
    3D模型的训练函数
    
    Args:
        model: 3D多输出模型
        dataloader: 3D数据加载器
        criterion: 3D多任务损失函数
        optimizer: 优化器
        device: 设备
        grad_clip_norm: 梯度裁剪范数
        
    Returns:
        tuple: (平均损失字典, 准确率字典)
    """
    model.train()
    
    # 损失累计
    total_losses = {'total': 0.0, 'return': 0.0, 'sharpe': 0.0, 'drawdown': 0.0}
    
    # 准确率累计（Top-K）
    correct_k = {
        'return': {1: 0, 2: 0, 3: 0},
        'sharpe': {1: 0, 2: 0, 3: 0},
        'drawdown': {1: 0, 2: 0, 3: 0}
    }
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training 3D", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        daily_data = batch['daily'].to(device)
        weekly_data = batch['weekly'].to(device)
        monthly_data = batch['monthly'].to(device)
        
        # 3D标签
        labels_3d = {}
        for dim in ['return', 'sharpe', 'drawdown']:
            labels_3d[dim] = batch['labels_3d'][dim].to(device)
        
        # 1. 前向传播
        outputs = model(daily_data, weekly_data, monthly_data)
        
        # 2. 详细的数值检查
        if _check_for_anomalies(outputs, labels_3d, batch_idx):
            continue  # 跳过有问题的batch
        
        # 3. 计算损失
        losses = criterion(outputs, labels_3d)
        
        # 检查损失有效性
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            if config.DEBUG_MODE:
                logger.error(f"Batch {batch_idx}: 检测到NaN/Inf损失，跳过...")
                _print_debug_info(outputs, labels_3d, losses)
            continue
        
        # 4. 反向传播
        optimizer.zero_grad()
        losses['total'].backward()
        
        # 5. 梯度裁剪
        if grad_clip_norm > 0:
            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            # if grad_norm > grad_clip_norm * 1.5:  # 只在梯度很大时打印
            #     print(f"Gradient clipped: norm was {grad_norm:.4f}")
        
        # 6. 检查梯度
        if _check_gradients(model):
            continue  # 跳过有问题的梯度
        
        # 7. 更新权重
        optimizer.step()
        
        # 8. 累计损失和准确率
        batch_size = daily_data.size(0)
        total_samples += batch_size
        
        for key, loss_val in losses.items():
            total_losses[key] += loss_val.item()
        
        # 计算Top-K准确率
        for dim in ['return', 'sharpe', 'drawdown']:
            _update_topk_accuracy(outputs[dim], labels_3d[dim], correct_k[dim])
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': losses['total'].item(),
            'ret_acc': correct_k['return'][1] / total_samples,
            'sha_acc': correct_k['sharpe'][1] / total_samples,
            'dd_acc': correct_k['drawdown'][1] / total_samples
        })
    
    # 计算平均损失和准确率
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    accuracies = {}
    for dim in ['return', 'sharpe', 'drawdown']:
        accuracies[dim] = {f'acc@{k}': v / total_samples for k, v in correct_k[dim].items()}
    
    return avg_losses, accuracies

def evaluate_3d(model, dataloader, criterion, device):
    """
    3D模型的评估函数
    """
    model.eval()
    
    total_losses = {'total': 0.0, 'return': 0.0, 'sharpe': 0.0, 'drawdown': 0.0}
    correct_k = {
        'return': {1: 0, 2: 0, 3: 0},
        'sharpe': {1: 0, 2: 0, 3: 0},
        'drawdown': {1: 0, 2: 0, 3: 0}
    }
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating 3D", leave=False)
        
        for batch in progress_bar:
            daily_data = batch['daily'].to(device)
            weekly_data = batch['weekly'].to(device)
            monthly_data = batch['monthly'].to(device)
            
            labels_3d = {}
            for dim in ['return', 'sharpe', 'drawdown']:
                labels_3d[dim] = batch['labels_3d'][dim].to(device)
            
            # 前向传播
            outputs = model(daily_data, weekly_data, monthly_data)
            
            # 计算损失
            losses = criterion(outputs, labels_3d)
            
            # 累计统计
            batch_size = daily_data.size(0)
            total_samples += batch_size
            
            for key, loss_val in losses.items():
                total_losses[key] += loss_val.item()
            
            # 计算准确率
            for dim in ['return', 'sharpe', 'drawdown']:
                _update_topk_accuracy(outputs[dim], labels_3d[dim], correct_k[dim])
    
    # 计算平均值
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    accuracies = {}
    for dim in ['return', 'sharpe', 'drawdown']:
        accuracies[dim] = {f'acc@{k}': v / total_samples for k, v in correct_k[dim].items()}
    
    return avg_losses, accuracies

def _check_for_anomalies(outputs, labels_3d, batch_idx):
    """检查输出和标签中的异常值"""
    has_anomaly = False
    
    for dim in ['return', 'sharpe', 'drawdown']:
        # 检查输出
        if torch.isnan(outputs[dim]).any() or torch.isinf(outputs[dim]).any():
            if config.DEBUG_MODE:
                logger.error(f"Batch {batch_idx}: {dim} 输出包含NaN/Inf")
            has_anomaly = True
        
        # 检查标签
        if torch.isnan(labels_3d[dim]).any() or torch.isinf(labels_3d[dim]).any():
            if config.DEBUG_MODE:
                logger.error(f"Batch {batch_idx}: {dim} 标签包含NaN/Inf")
            has_anomaly = True
        
        # 检查标签概率和
        prob_sums = labels_3d[dim].sum(dim=1)
        if (torch.abs(prob_sums - 1.0) > 1e-5).any():
            if config.DEBUG_MODE:
                logger.warning(f"Batch {batch_idx}: {dim} 标签概率和不正常")
            has_anomaly = True
    
    return has_anomaly

def _check_gradients(model):
    """检查梯度中的异常值"""
    has_nan_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                if config.DEBUG_MODE:
                    logger.error(f"NaN/Inf梯度在 {name} 中")
                has_nan_grad = True
    
    return has_nan_grad

def _update_topk_accuracy(outputs, targets, correct_dict):
    """更新Top-K准确率"""
    # 获取真实类别（最大概率对应的类别）
    _, true_classes = torch.max(targets, 1)
    
    # 获取Top-K预测
    _, top_k_preds = torch.topk(outputs, 3, dim=1)
    
    # 计算Top-K准确率
    for k in [1, 2, 3]:
        correct_dict[k] += (top_k_preds[:, :k] == true_classes.unsqueeze(1)).any(dim=1).sum().item()

def _print_debug_info(outputs, labels_3d, losses):
    """打印详细的调试信息（仅在DEBUG模式下）"""
    if not config.DEBUG_MODE:
        return
        
    logger.debug("=== 调试信息 ===")
    logger.debug("输出:")
    for dim, output in outputs.items():
        logger.debug(f"  {dim}: min={output.min().item():.6f}, max={output.max().item():.6f}")
    
    logger.debug("标签:")
    for dim, label in labels_3d.items():
        logger.debug(f"  {dim}: min={label.min().item():.6f}, max={label.max().item():.6f}")
    
    logger.debug("损失:")
    for dim, loss in losses.items():
        logger.debug(f"  {dim}: {loss.item():.6f}")

def format_3d_results(losses, accuracies):
    """
    格式化3D训练结果用于打印
    """
    # 格式化损失
    loss_str = f"Total: {losses['total']:.6f}"
    for dim in ['return', 'sharpe', 'drawdown']:
        if dim in losses:
            loss_str += f", {dim.capitalize()}: {losses[dim]:.6f}"
    
    # 格式化准确率
    acc_parts = []
    for dim in ['return', 'sharpe', 'drawdown']:
        if dim in accuracies:
            dim_accs = ", ".join([f"{k}: {v:.4f}" for k, v in accuracies[dim].items()])
            acc_parts.append(f"{dim.capitalize()}[{dim_accs}]")
    
    acc_str = " | ".join(acc_parts)
    
    return loss_str, acc_str

def test_3d_engine():
    """
    测试3D训练引擎
    """
    logger.info("=== 测试3D训练引擎 ===")
    
    # 这里只是一个简单的测试框架
    # 实际测试需要真实的模型、数据和损失函数
    logger.info("3D训练引擎功能:")
    if config.DEBUG_MODE:
        logger.debug("  多维度损失计算")
        logger.debug("  多维度准确率统计")
        logger.debug("  数值稳定性检查")
        logger.debug("  梯度异常检测")
        logger.debug("  详细的调试信息")
    else:
        logger.info("  完整的多维度训练支持")
    
    # 测试格式化函数
    test_losses = {'total': 1.234, 'return': 0.456, 'sharpe': 0.789, 'drawdown': 0.321}
    test_accs = {
        'return': {'acc@1': 0.3, 'acc@2': 0.5, 'acc@3': 0.7},
        'sharpe': {'acc@1': 0.25, 'acc@2': 0.45, 'acc@3': 0.65},
        'drawdown': {'acc@1': 0.35, 'acc@2': 0.55, 'acc@3': 0.75}
    }
    
    loss_str, acc_str = format_3d_results(test_losses, test_accs)
    if config.DEBUG_MODE:
        logger.debug(f"示例输出格式:")
        logger.debug(f"Loss: {loss_str}")
        logger.debug(f"Accs: {acc_str}")
    
    logger.info("3D训练引擎测试通过")

if __name__ == '__main__':
    test_3d_engine()