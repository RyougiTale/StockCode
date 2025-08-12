import torch
from tqdm import tqdm

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
        
        # 2. 计算损失
        loss = criterion(outputs, soft_labels)
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 4. 更新权重
        optimizer.step()

        # 累计损失和Top-K准确率
        total_loss += loss.item()
        
        _, true_classes = torch.max(soft_labels.data, 1)
        _, top_k_preds = torch.topk(outputs.data, 3, dim=1)
        
        total_samples += soft_labels.size(0)
        for k in correct_k.keys():
            # 检查真实类别是否存在于Top-K预测中
            correct_k[k] += (top_k_preds[:, :k] == true_classes.unsqueeze(1)).any(dim=1).sum().item()
        
        progress_bar.set_postfix(loss=loss.item())

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