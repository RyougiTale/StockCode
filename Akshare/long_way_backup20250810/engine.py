import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm
from . import config
from .logger_config import get_logger

# 获取日志记录器
logger = get_logger(__name__)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    对模型进行一个epoch的训练（新版：KLDivLoss + Top-K Accuracy）。
    """
    model.train()
    total_loss = 0.0
    correct_k = {1: 0, 2: 0, 3: 0}
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        daily_data = batch['daily'].to(device)
        weekly_data = batch['weekly'].to(device)
        monthly_data = batch['monthly'].to(device)
        soft_labels = batch['label'].to(device)

        # 1. 前向传播
        outputs = model(daily_data, weekly_data, monthly_data)
        
        # 2. 计算损失前的详细检查
        if config.DEBUG_MODE and config.ENABLE_DATA_VALIDATION:
            batch_idx = progress_bar.n
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.error(f"Batch {batch_idx}: 模型输出包含NaN或Inf!")
                logger.error(f"输出形状: {outputs.shape}, NaN数量: {torch.isnan(outputs).sum().item()}, Inf数量: {torch.isinf(outputs).sum().item()}")
                
            if torch.isnan(soft_labels).any() or torch.isinf(soft_labels).any():
                logger.error(f"Batch {batch_idx}: 软标签包含NaN或Inf!")
        elif not config.DEBUG_MODE:
            # 生产模式下只做关键检查
            if torch.isnan(outputs).any() or torch.isinf(outputs).any() or torch.isnan(soft_labels).any() or torch.isinf(soft_labels).any():
                logger.error("训练中发现数值异常，建议开启DEBUG_MODE进行详细检查")
            
        loss = criterion(outputs, soft_labels)

        if torch.isnan(loss) or torch.isinf(loss):
            if config.DEBUG_MODE:
                batch_idx = progress_bar.n
                logger.error(f"Batch {batch_idx}: 检测到NaN/Inf损失")
                logger.error(f"模型输出(log-probs): 形状={outputs.shape}, 范围=[{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                logger.error(f"目标软标签(probs): 形状={soft_labels.shape}, 范围=[{soft_labels.min().item():.6f}, {soft_labels.max().item():.6f}]")
                logger.error(f"输入数据范围: 日线=[{daily_data.min().item():.6f}, {daily_data.max().item():.6f}], 周线=[{weekly_data.min().item():.6f}, {weekly_data.max().item():.6f}], 月线=[{monthly_data.min().item():.6f}, {monthly_data.max().item():.6f}]")
            
            raise RuntimeError("检测到NaN/Inf损失，停止训练进行诊断")
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 4. 梯度裁剪（防止梯度爆炸）
        # if hasattr(config, 'GRAD_CLIP_NORM'):
        #     grad_norm = nn_utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        #     if grad_norm > config.GRAD_CLIP_NORM:
        #         print(f"Gradient clipped: norm was {grad_norm:.4f}")
        
        # 5. 检查梯度
        if config.ENABLE_DATA_VALIDATION:
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.error("检测到NaN/Inf梯度")
                        raise RuntimeError("检测到NaN/Inf梯度")
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            if config.DEBUG_MODE and total_grad_norm > 10.0:  # 梯度太大时警告
                logger.warning(f"梯度范数较大: {total_grad_norm:.4f}")
        
        # 6. 更新权重
        optimizer.step()

        # 累计损失和Top-K准确率
        total_loss += loss.item()
        
        _, true_classes = torch.max(soft_labels.data, 1)
        _, top_k_preds = torch.topk(outputs.data, 3, dim=1)
        
        total_samples += soft_labels.size(0)
        for k in correct_k.keys():
            # 检查真实类别是否存在于Top-K预测中
            correct_k[k] += (top_k_preds[:, :k] == true_classes.unsqueeze(1)).any(dim=1).sum().item()
        
        # 每个批次的进度显示（简化版）
        if config.DEBUG_MODE:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", grad_norm=f"{total_grad_norm:.2f}" if 'total_grad_norm' in locals() else "N/A")
        else:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracies = {f'acc@{k}': v / total_samples for k, v in correct_k.items()}
    
    return avg_loss, accuracies

def evaluate(model, dataloader, criterion, device):
    """
    在验证集上评估模型（新版：KLDivLoss + Top-K Accuracy）。
    """
    model.eval()
    total_loss = 0.0
    correct_k = {1: 0, 2: 0, 3: 0}
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            daily_data = batch['daily'].to(device)
            weekly_data = batch['weekly'].to(device)
            monthly_data = batch['monthly'].to(device)
            soft_labels = batch['label'].to(device)

            outputs = model(daily_data, weekly_data, monthly_data)
            loss = criterion(outputs, soft_labels)

            total_loss += loss.item()

            _, true_classes = torch.max(soft_labels.data, 1)
            _, top_k_preds = torch.topk(outputs.data, 3, dim=1)
            
            total_samples += soft_labels.size(0)
            for k in correct_k.keys():
                correct_k[k] += (top_k_preds[:, :k] == true_classes.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracies = {f'acc@{k}': v / total_samples for k, v in correct_k.items()}
    
    return avg_loss, accuracies