import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    对模型进行一个epoch的训练。
    """
    model.train()  # 将模型设置为训练模式
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 使用tqdm来显示进度条
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # 通过解包获取数据，使其对字典和元组都兼容
        if isinstance(batch, dict):
            daily_data = batch['daily'].to(device)
            weekly_data = batch['weekly'].to(device)
            monthly_data = batch['monthly'].to(device)
            labels = batch['label'].to(device)
        else:
            daily_data, weekly_data, monthly_data, labels = [d.to(device) for d in batch]

        # 1. 前向传播
        outputs = model(daily_data, weekly_data, monthly_data)
        
        # 2. 计算损失
        loss = criterion(outputs, labels)
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4. 更新权重
        optimizer.step()

        # 累计损失和正确率
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # 更新进度条信息
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """
    在验证集上评估模型。
    """
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 在评估阶段不计算梯度
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            if isinstance(batch, dict):
                daily_data = batch['daily'].to(device)
                weekly_data = batch['weekly'].to(device)
                monthly_data = batch['monthly'].to(device)
                labels = batch['label'].to(device)
            else:
                daily_data, weekly_data, monthly_data, labels = [d.to(device) for d in batch]

            outputs = model(daily_data, weekly_data, monthly_data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy